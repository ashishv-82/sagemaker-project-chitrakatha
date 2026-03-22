"""Flow B: Silver corpus chunks → RAFT training data → S3 Gold.

Why: This script implements the RAFT (Retrieval-Augmented Fine-Tuning) data
     synthesis strategy. For each golden chunk, it samples 2 distractor chunks
     from different entities/publishers, then calls Bedrock Claude 3.5 Sonnet
     to generate 3 bilingual Q&A pairs with explicit chain-of-thought reasoning.

     The resulting Gold JSONL is what the QLoRA fine-tuning step trains on.
     Because the model sees a golden document + distractors during training,
     it learns to select relevant context at inference time — exactly the skill
     it needs when querying the live S3 Vectors index.

Output JSONL schema (one record per Q&A pair):
    {
        "id": "<uuid4>",
        "question_en": "...",
        "question_hi": "... (Devanagari)",
        "golden_chunk": "... (the source passage)",
        "distractor_chunks": ["...", "..."],
        "chain_of_thought": "The question asks about X. Document 1 is relevant because...",
        "answer_en": "... (grounded in golden_chunk only)",
        "answer_hi": "... (Devanagari, grounded in golden_chunk only)",
        "source_chunk_id": "<uuid4>",
        "source_entity": "Nagraj",
        "publisher": "Raj Comics",
        "language_pair": "en-hi"
    }

Constraints:
    - The Claude system prompt explicitly instructs grounding-only answers.
    - Distractors are sampled from a different entity/publisher to maximise
      subject-matter contrast and training signal.
    - Exponential backoff retry on Bedrock throttling (3 attempts).
    - API key retrieved from Secrets Manager at runtime — never from env.
"""

from __future__ import annotations

import json
import logging
import random
import sys
import time
import uuid
from typing import Any, Final

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from chitrakatha.config import Settings
from chitrakatha.exceptions import BedrockSynthesisError, DataIngestionError

logger = logging.getLogger(__name__)

_CLAUDE_MODEL_ID: Final[str] = "ap.anthropic.claude-haiku-4-5-20251001-v1:0"
_SILVER_TRAINING_PREFIX: Final[str] = "training/"
_GOLD_TRAINING_PREFIX: Final[str] = "training-pairs/"
_QA_PAIRS_PER_CHUNK: Final[int] = 3
_NUM_DISTRACTORS: Final[int] = 2
_MAX_RETRIES: Final[int] = 3
_RETRY_BASE_DELAY_SEC: Final[float] = 2.0

# RAFT synthesis prompt template.
_RAFT_SYSTEM_PROMPT: Final[str] = (
    "You are an expert on Indian comic books. Your task is to generate bilingual "
    "training data for a retrieval-augmented language model.\n\n"
    "You will receive:\n"
    "  1. A GOLDEN DOCUMENT containing factual information.\n"
    "  2. Two DISTRACTOR DOCUMENTS that are about different characters or topics.\n\n"
    "Generate exactly {n_pairs} bilingual Q&A training examples as a JSON array. "
    "Each element must have these exact fields:\n"
    "  question_en, question_hi, chain_of_thought, answer_en, answer_hi\n\n"
    "Rules:\n"
    "  - Questions must be answerable ONLY from the golden document.\n"
    "  - chain_of_thought must explicitly state which document contains the answer "
    "and why the distractors are NOT relevant.\n"
    "  - Answers must be grounded strictly in the golden document — no hallucination.\n"
    "  - question_hi and answer_hi must be in Devanagari script (Hindi).\n"
    "  - Output ONLY valid JSON — no markdown, no explanation outside JSON."
)

_RAFT_USER_TEMPLATE: Final[str] = (
    "[GOLDEN DOCUMENT]:\n{golden}\n\n"
    "[DISTRACTOR DOCUMENT 1]:\n{distractor_1}\n\n"
    "[DISTRACTOR DOCUMENT 2]:\n{distractor_2}\n\n"
    "Generate {n_pairs} bilingual Q&A training pairs as a JSON array."
)


def _get_secret(secret_name: str, region: str) -> str:
    """Retrieve a secret value from AWS Secrets Manager at runtime.

    Args:
        secret_name: The secret name (e.g. ``chitrakatha/synthetic_data_api_key``).
        region: AWS region.

    Returns:
        The raw secret string value.

    Raises:
        DataIngestionError: If the secret cannot be retrieved.
    """
    client = boto3.client("secretsmanager", region_name=region)
    try:
        response = client.get_secret_value(SecretId=secret_name)
        return response.get("SecretString", "")
    except (BotoCoreError, ClientError) as exc:
        raise DataIngestionError(
            f"Cannot retrieve secret '{secret_name}': {exc}"
        ) from exc


def _call_claude(
    bedrock_client: boto3.client,
    golden_chunk: str,
    distractor_chunks: list[str],
    n_pairs: int = _QA_PAIRS_PER_CHUNK,
) -> list[dict[str, Any]]:
    """Call Claude 3.5 Sonnet to generate RAFT Q&A pairs.

    Args:
        bedrock_client: Pre-built ``bedrock-runtime`` boto3 client.
        golden_chunk: The factual source passage.
        distractor_chunks: Exactly 2 irrelevant passages.
        n_pairs: Number of Q&A pairs to generate per chunk.

    Returns:
        List of Q&A dicts parsed from Claude's JSON response.

    Raises:
        BedrockSynthesisError: After all retries or on non-retryable error.
    """
    system_prompt = _RAFT_SYSTEM_PROMPT.format(n_pairs=n_pairs)
    user_message = _RAFT_USER_TEMPLATE.format(
        golden=golden_chunk,
        distractor_1=distractor_chunks[0],
        distractor_2=distractor_chunks[1],
        n_pairs=n_pairs,
    )

    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_message}],
    })

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId=_CLAUDE_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=body,
            )
            raw = json.loads(response["body"].read())
            text = raw["content"][0]["text"].strip()
            return json.loads(text)

        except json.JSONDecodeError as exc:
            logger.warning("Claude returned non-JSON output (attempt %d): %s", attempt, exc)
            if attempt == _MAX_RETRIES:
                raise BedrockSynthesisError(
                    "Claude failed to produce valid JSON after all retries."
                ) from exc

        except ClientError as exc:
            code = exc.response["Error"]["Code"]
            if code in {"ThrottlingException", "ServiceUnavailableException"} and attempt < _MAX_RETRIES:
                delay = _RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                logger.warning("Claude throttled (attempt %d). Retrying in %.1fs.", attempt, delay)
                time.sleep(delay)
                continue
            raise BedrockSynthesisError(
                f"Bedrock Claude call failed [{code}]: {exc}"
            ) from exc

    raise BedrockSynthesisError("Claude synthesis failed after all retries.")


def _list_silver_training_chunks(
    s3_client: boto3.client,
    bucket: str,
) -> list[dict[str, Any]]:
    """Load all training chunks from S3 Silver /training/ as a flat list.

    Args:
        s3_client: Boto3 S3 client.
        bucket: Silver bucket name.

    Returns:
        List of chunk dicts with at least ``text``, ``source_document``,
        ``source_entity``, ``publisher``, and ``language`` fields.
    """
    chunks: list[dict[str, Any]] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=_SILVER_TRAINING_PREFIX):
        for obj in page.get("Contents", []):
            response = s3_client.get_object(Bucket=bucket, Key=obj["Key"])
            for line in response["Body"].read().decode("utf-8").splitlines():
                line = line.strip()
                if line:
                    try:
                        chunks.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("Skipping bad JSON line in %s", obj["Key"])
    return chunks


def run(settings: Settings) -> int:
    """Execute the full Flow B RAFT synthesis pipeline.

    Args:
        settings: Pydantic settings from environment.

    Returns:
        Number of chunks that failed processing.
    """
    s3_client = boto3.client("s3", region_name=settings.aws_region)
    bedrock_client = boto3.client("bedrock-runtime", region_name=settings.aws_region)

    all_chunks = _list_silver_training_chunks(s3_client, settings.s3_silver_bucket)
    logger.info("Loaded %d training chunk(s) from Silver /training/.", len(all_chunks))

    if len(all_chunks) < _NUM_DISTRACTORS + 1:
        raise DataIngestionError(
            f"Need at least {_NUM_DISTRACTORS + 1} chunks to generate RAFT data; "
            f"found {len(all_chunks)}."
        )

    errors = 0
    total_pairs = 0
    output_lines: list[str] = []

    for i, golden in enumerate(all_chunks):
        # Sample distractors from a different entity/publisher where possible.
        golden_entity = golden.get("source_entity", "")
        golden_publisher = golden.get("publisher", "")

        candidates = [
            c for c in all_chunks
            if c is not golden
            and (c.get("source_entity") != golden_entity or c.get("publisher") != golden_publisher)
        ]

        # Fall back to any non-self chunk if cross-entity pool is too small.
        if len(candidates) < _NUM_DISTRACTORS:
            candidates = [c for c in all_chunks if c is not golden]

        distractor_chunks_data = random.sample(candidates, _NUM_DISTRACTORS)
        distractor_texts = [d["text"] for d in distractor_chunks_data]

        try:
            pairs = _call_claude(
                bedrock_client,
                golden_chunk=golden["text"],
                distractor_chunks=distractor_texts,
            )
        except (BedrockSynthesisError, DataIngestionError) as exc:
            logger.error("Failed chunk %d ('%s'): %s", i, golden.get("source_document"), exc)
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
                "source_entity": golden_entity,
                "publisher": golden_publisher,
                "language_pair": "en-hi",
            }
            output_lines.append(json.dumps(record, ensure_ascii=False))
            total_pairs += 1

    # Write all pairs to S3 Gold /training-pairs/.
    if output_lines:
        output_key = f"{_GOLD_TRAINING_PREFIX}raft_pairs_{uuid.uuid4().hex[:8]}.jsonl"
        gold_bytes = "\n".join(output_lines).encode("utf-8")

        try:
            s3_client.put_object(
                Bucket=settings.s3_gold_bucket,
                Key=output_key,
                Body=gold_bytes,
                ContentType="application/x-ndjson; charset=utf-8",
                ServerSideEncryption="aws:kms",
                SSEKMSKeyId=settings.kms_key_arn,
            )
            logger.info(
                "Wrote %d RAFT pair(s) to s3://%s/%s",
                total_pairs, settings.s3_gold_bucket, output_key,
            )
        except (BotoCoreError, ClientError) as exc:
            raise DataIngestionError(
                f"Failed to write RAFT pairs to S3 Gold: {exc}"
            ) from exc

    logger.info(
        "Flow B complete. Pairs generated: %d. Errors: %d chunk(s).",
        total_pairs, errors,
    )
    return errors


def main() -> None:
    """CLI / SageMaker Processing Job entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    error_count = run(settings)
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
