"""SageMaker Processing step: Flow B — RAFT training pair synthesis.

Why: This step wraps synthesize_training_pairs.py for execution as a
     SageMaker Processing Job. It reads the training JSONL from the Silver
     /training/ channel (produced by preprocessing.py) and calls Bedrock
     Claude 3.5 Sonnet to generate bilingual RAFT Q&A pairs.

     This step runs in parallel with embed_and_index.py (Flow A) inside
     the SageMaker Pipeline DAG — both depend only on preprocessing.py.

Input channel (SageMaker):
    /opt/ml/processing/input/training — training.jsonl from preprocessing step
Output channels (SageMaker):
    /opt/ml/processing/output/gold/train — 90% of pairs, written to S3 Gold /training-pairs/train/
    /opt/ml/processing/output/gold/eval  — 10% of pairs (held-out), written to S3 Gold /training-pairs/eval/

Metrics logged to SageMaker Experiments:
    - ``raft_pairs_generated``: total Q&A pairs produced
    - ``synthesis_errors``: chunks that failed Claude synthesis
    - ``bedrock_tokens_used``: total input+output tokens (from Claude response)

Constraints:
    - Exits with code 1 if synthesis_errors > 0 (SageMaker marks step FAILED).
    - Training JSONL path is injected by the Pipeline DAG, not hardcoded.
"""

from __future__ import annotations

import subprocess
import sys

# These packages are not pre-installed in the ScriptProcessor container.
# Install before any chitrakatha imports that depend on them.
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "pydantic>=2.5.0", "pydantic-settings>=2.2.0", "pytz",
    "--quiet",
])

import json
import logging
import os
import random
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Final

import boto3
from botocore.exceptions import ClientError

from chitrakatha.config import Settings
from chitrakatha.exceptions import BedrockSynthesisError, DataIngestionError
from chitrakatha.monitoring.experiments import log_metrics

logger = logging.getLogger(__name__)

# SageMaker Processing Job channel mount points.
INPUT_TRAINING_PATH = Path("/opt/ml/processing/input/training/training.jsonl")
OUTPUT_GOLD_DIR = Path("/opt/ml/processing/output/gold")
OUTPUT_TRAIN_DIR = OUTPUT_GOLD_DIR / "train"
OUTPUT_EVAL_DIR = OUTPUT_GOLD_DIR / "eval"

# Fraction of synthesised pairs held out for evaluation (not seen during training).
_EVAL_SPLIT: Final[float] = 0.10

_CLAUDE_MODEL_ID: Final[str] = "anthropic.claude-3-5-sonnet-20241022-v2:0"
_QA_PAIRS_PER_CHUNK: Final[int] = 3
_NUM_DISTRACTORS: Final[int] = 2
_MAX_RETRIES: Final[int] = 3

_RAFT_SYSTEM_PROMPT: Final[str] = (
    "You are an expert on Indian comic books. Generate exactly {n_pairs} bilingual "
    "Q&A training examples as a JSON array. Each element must have: "
    "question_en, question_hi, chain_of_thought, answer_en, answer_hi. "
    "Rules: questions answerable only from the golden document; "
    "chain_of_thought explicitly identifies the relevant document and why distractors are not; "
    "answers grounded strictly in the golden document; "
    "question_hi and answer_hi in Devanagari. Output ONLY valid JSON."
)

_RAFT_USER_TEMPLATE: Final[str] = (
    "[GOLDEN DOCUMENT]:\n{golden}\n\n"
    "[DISTRACTOR 1]:\n{d1}\n\n"
    "[DISTRACTOR 2]:\n{d2}\n\n"
    "Generate {n_pairs} bilingual RAFT Q&A pairs as a JSON array."
)


def _call_claude(
    bedrock_client: boto3.client,
    golden: str,
    distractors: list[str],
    total_tokens: list[int],
) -> list[dict[str, Any]]:
    """Call Claude 3.5 Sonnet to synthesise RAFT pairs. Updates total_tokens in-place."""
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": _RAFT_SYSTEM_PROMPT.format(n_pairs=_QA_PAIRS_PER_CHUNK),
        "messages": [{
            "role": "user",
            "content": _RAFT_USER_TEMPLATE.format(
                golden=golden, d1=distractors[0], d2=distractors[1],
                n_pairs=_QA_PAIRS_PER_CHUNK,
            ),
        }],
    })

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = bedrock_client.invoke_model(
                modelId=_CLAUDE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            raw = json.loads(resp["body"].read())
            usage = raw.get("usage", {})
            total_tokens[0] += usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            parsed = json.loads(raw["content"][0]["text"].strip())
            # Claude sometimes wraps the array in a dict (e.g. {"pairs": [...]}).
            # Unwrap to always return a list of dicts.
            if isinstance(parsed, dict):
                parsed = next(
                    (v for v in parsed.values() if isinstance(v, list)), []
                )
            return parsed

        except json.JSONDecodeError as exc:
            if attempt == _MAX_RETRIES:
                raise BedrockSynthesisError("Claude returned invalid JSON after retries.") from exc

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in {"ThrottlingException", "ServiceUnavailableException"} and attempt < _MAX_RETRIES:
                time.sleep(2.0 * (2 ** (attempt - 1)))
                continue
            raise BedrockSynthesisError(f"Bedrock error [{code}]: {exc}") from exc

    raise BedrockSynthesisError("Claude synthesis failed after all retries.")


def run(settings: Settings, experiment_run_name: str | None = None) -> int:
    """Execute Flow B RAFT synthesis from the Silver training channel.

    Args:
        settings: Runtime settings.
        experiment_run_name: SageMaker Experiments run for metric logging.

    Returns:
        Error count (0 = success).
    """
    if not INPUT_TRAINING_PATH.exists():
        logger.error("Training input not found: %s", INPUT_TRAINING_PATH)
        return 1

    bedrock_client = boto3.client("bedrock-runtime", region_name=settings.aws_region)
    s3_client = boto3.client("s3", region_name=settings.aws_region)

    all_chunks: list[dict[str, Any]] = []
    for line in INPUT_TRAINING_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                all_chunks.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping bad JSON line in training input.")

    logger.info("Loaded %d training chunks.", len(all_chunks))

    if len(all_chunks) < _NUM_DISTRACTORS + 1:
        raise DataIngestionError(
            f"Need >= {_NUM_DISTRACTORS + 1} chunks for RAFT; found {len(all_chunks)}."
        )

    errors = 0
    total_pairs = 0
    total_tokens = [0]
    output_lines: list[str] = []

    for i, golden in enumerate(all_chunks):
        # Prefer distractors from a different source document so the model must
        # distinguish across documents, not just within one.  source_document is
        # always present (written by preprocessing.py).
        golden_doc = golden.get("source_document", "")

        candidates = [
            c for c in all_chunks
            if c is not golden and c.get("source_document", "") != golden_doc
        ] or [c for c in all_chunks if c is not golden]

        distractors = random.sample(candidates, min(_NUM_DISTRACTORS, len(candidates)))
        distractor_texts = [d["text"] for d in distractors]

        try:
            pairs = _call_claude(bedrock_client, golden["text"], distractor_texts, total_tokens)
        except (BedrockSynthesisError, DataIngestionError) as exc:
            logger.error("Chunk %d failed: %s", i, exc)
            errors += 1
            continue

        for pair in pairs:
            record = {
                "id": str(uuid.uuid4()),
                "question_en": pair.get("question_en", ""),
                "question_hi": pair.get("question_hi", ""),
                "golden_chunk": golden["text"],
                "distractor_chunks": distractor_texts,
                "chain_of_thought": pair.get("chain_of_thought", ""),
                "answer_en": pair.get("answer_en", ""),
                "answer_hi": pair.get("answer_hi", ""),
                "source_chunk_id": golden.get("chunk_id", ""),
                "source_document": golden_doc,
                "language_pair": "en-hi",
            }
            output_lines.append(json.dumps(record, ensure_ascii=False))
            total_pairs += 1

    # Split into train (90%) / eval (10%) — shuffle first to avoid entity bias.
    random.shuffle(output_lines)
    n_eval = max(1, int(len(output_lines) * _EVAL_SPLIT))
    eval_lines = output_lines[:n_eval]
    train_lines = output_lines[n_eval:]

    run_id = uuid.uuid4().hex[:8]
    OUTPUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_TRAIN_DIR / f"raft_pairs_{run_id}.jsonl"
    eval_path = OUTPUT_EVAL_DIR / f"raft_pairs_eval_{run_id}.jsonl"
    train_path.write_text("\n".join(train_lines), encoding="utf-8")
    eval_path.write_text("\n".join(eval_lines), encoding="utf-8")
    logger.info(
        "Wrote %d train pair(s) to %s, %d eval pair(s) to %s.",
        len(train_lines), train_path, len(eval_lines), eval_path,
    )

    if experiment_run_name:
        log_metrics(
            run_name=experiment_run_name,
            metrics={
                "raft_pairs_generated": total_pairs,
                "synthesis_errors": errors,
                "bedrock_tokens_used": total_tokens[0],
            },
        )

    return errors


def main() -> None:
    """SageMaker Processing Job entry point."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    settings = Settings()
    experiment_run_name = os.environ.get("SAGEMAKER_EXPERIMENT_RUN")
    sys.exit(1 if run(settings, experiment_run_name) > 0 else 0)


if __name__ == "__main__":
    main()
