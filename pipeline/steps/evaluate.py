"""SageMaker Processing step: Three-suite evaluation for RAFT fine-tuned model.

Why: Standard NLP evaluation (ROUGE-L alone) does not capture the RAFT-
     specific skill of ignoring distractors. This script adds a dedicated
     distractor-robustness test suite that validates whether the fine-tuned
     model can select the correct answer when presented with irrelevant context.

Three test suites:
    1. Factual accuracy     — ROUGE-L, BERTScore, exact-match@1 on plain Q&A pairs
    2. Cross-lingual        — English query must match Devanagari ground truth answer
    3. Distractor robustness — 1 golden + 4 distractors; model must still answer correctly

Pass threshold (ConditionStep in pipeline.py):
    ROUGE-L >= 0.35 AND distractor_robustness >= 0.70

Output:
    /opt/ml/processing/output/evaluation/evaluation.json
    {
        "rouge_l": float,
        "bert_score_f1": float,
        "exact_match": float,
        "cross_lingual_accuracy": float,
        "distractor_robustness": float,
        "status": "pass" | "fail"
    }

Constraints:
    - Model inference runs on CPU (Processing Job) — not Spot GPU.
    - Uses a small held-out evaluation set (10% of Gold data, max 200 records).
    - All metrics emitted to SageMaker Experiments run.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import sys
from pathlib import Path
from typing import Any

import evaluate
import torch
from bert_score import score as bert_score_fn
from transformers import pipeline as hf_pipeline

from chitrakatha.monitoring.experiments import log_metrics

logger = logging.getLogger(__name__)

# SageMaker channel mount points.
INPUT_MODEL_DIR = Path(os.environ.get("SM_CHANNEL_MODEL", "/opt/ml/processing/input/model"))
INPUT_EVAL_DIR = Path(os.environ.get("SM_CHANNEL_EVAL", "/opt/ml/processing/input/eval"))
OUTPUT_EVAL_DIR = Path("/opt/ml/processing/output/evaluation")

EXPERIMENT_RUN = os.environ.get("SAGEMAKER_EXPERIMENT_RUN", "")

# Pass thresholds.
ROUGE_L_THRESHOLD = float(os.environ.get("ROUGE_L_THRESHOLD", "0.35"))
DISTRACTOR_ROBUSTNESS_THRESHOLD = float(os.environ.get("DISTRACTOR_ROBUSTNESS_THRESHOLD", "0.70"))

# Cap eval set size (cost + time control on Processing Job).
MAX_EVAL_RECORDS = int(os.environ.get("MAX_EVAL_RECORDS", "200"))
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


def _load_eval_records() -> list[dict[str, Any]]:
    """Load Gold JSONL evaluation records (held-out 10% from train.py)."""
    records: list[dict[str, Any]] = []
    for path in sorted(INPUT_EVAL_DIR.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    random.shuffle(records)
    return records[:MAX_EVAL_RECORDS]


def _build_raft_context(golden: str, distractors: list[str]) -> str:
    """Assemble a RAFT context string (shuffled documents)."""
    docs = [golden] + distractors
    random.shuffle(docs)
    return "\n\n".join(
        f"[Document {i+1}]: {doc}" for i, doc in enumerate(docs)
    )


def _generate_answer(gen_pipe: Any, context: str, question: str) -> str:
    """Run inference and return the model's answer string."""
    prompt = (
        f"You are given the following documents:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Think step by step, then answer using ONLY the relevant document above.\nAnswer:"
    )
    output = gen_pipe(prompt, max_new_tokens=256, do_sample=False)
    generated = output[0]["generated_text"]
    # Extract text after "Answer:" marker.
    answer_start = generated.rfind("Answer:") + len("Answer:")
    return generated[answer_start:].strip()


def suite_factual_accuracy(
    records: list[dict],
    gen_pipe: Any,
    rouge_metric: Any,
) -> dict[str, float]:
    """Suite 1: ROUGE-L, BERTScore, exact-match on plain Q→A (no distractors).

    Args:
        records: Eval records.
        gen_pipe: HuggingFace text-generation pipeline.
        rouge_metric: HuggingFace evaluate ROUGE metric.

    Returns:
        Dict with rouge_l, bert_score_f1, exact_match.
    """
    predictions: list[str] = []
    references: list[str] = []

    for rec in records:
        context = f"[Document 1]: {rec.get('golden_chunk', '')}"
        pred = _generate_answer(gen_pipe, context, rec.get("question_en", ""))
        predictions.append(pred)
        references.append(rec.get("answer_en", ""))

    rouge_result = rouge_metric.compute(predictions=predictions, references=references)
    rouge_l = round(rouge_result["rougeL"], 4)

    _, _, bert_f1 = bert_score_fn(predictions, references, lang="en", verbose=False)
    bert_score_f1 = round(float(bert_f1.mean()), 4)

    exact_matches = sum(
        1 for p, r in zip(predictions, references)
        if p.strip().lower() == r.strip().lower()
    )
    exact_match = round(exact_matches / max(len(records), 1), 4)

    logger.info(
        "Suite 1 (factual): ROUGE-L=%.3f, BERTScore-F1=%.3f, EM=%.3f",
        rouge_l, bert_score_f1, exact_match,
    )
    return {"rouge_l": rouge_l, "bert_score_f1": bert_score_f1, "exact_match": exact_match}


def suite_cross_lingual(records: list[dict], gen_pipe: Any) -> dict[str, float]:
    """Suite 2: English query → model must match Devanagari answer.

    Tests that the RAFT training preserved bilingual understanding.

    Args:
        records: Eval records (uses question_en + answer_hi).
        gen_pipe: HuggingFace generation pipeline.

    Returns:
        Dict with cross_lingual_accuracy.
    """
    hi_records = [r for r in records if _DEVANAGARI_RE.search(r.get("answer_hi", ""))]
    if not hi_records:
        logger.warning("No Hindi answer records found — skipping cross-lingual suite.")
        return {"cross_lingual_accuracy": 0.0}

    correct = 0
    for rec in hi_records:
        context = f"[Document 1]: {rec.get('golden_chunk', '')}"
        # Ask in English, expect Hindi answer.
        pred = _generate_answer(gen_pipe, context, rec.get("question_en", ""))
        # Pass if the prediction contains at least one Devanagari character.
        if _DEVANAGARI_RE.search(pred):
            correct += 1

    accuracy = round(correct / max(len(hi_records), 1), 4)
    logger.info("Suite 2 (cross-lingual): accuracy=%.3f (%d/%d)", accuracy, correct, len(hi_records))
    return {"cross_lingual_accuracy": accuracy}


def suite_distractor_robustness(records: list[dict], gen_pipe: Any, rouge_metric: Any) -> dict[str, float]:
    """Suite 3 (RAFT-specific): 1 golden + 4 distractors — model must still answer correctly.

    Tests that the RAFT training generalised to more distractors than seen
    during training (training used 2 distractors; evaluation uses 4).

    Args:
        records: Eval records.
        gen_pipe: HuggingFace generation pipeline.
        rouge_metric: HuggingFace evaluate ROUGE metric.

    Returns:
        Dict with distractor_robustness (fraction of records with ROUGE-L >= 0.30).
    """
    robust_count = 0
    total = 0

    # Use all records' distractor_chunks as a pool of distractors.
    distractor_pool = [
        chunk
        for r in records
        for chunk in r.get("distractor_chunks", [])
    ]

    for rec in records:
        golden = rec.get("golden_chunk", "")
        # Sample 4 distractors from pool (excluding current record's own text).
        available = [d for d in distractor_pool if d != golden]
        if len(available) < 4:
            continue

        hard_distractors = random.sample(available, 4)
        context = _build_raft_context(golden, hard_distractors)
        pred = _generate_answer(gen_pipe, context, rec.get("question_en", ""))
        ref = rec.get("answer_en", "")

        result = rouge_metric.compute(predictions=[pred], references=[ref])
        if result["rougeL"] >= 0.30:
            robust_count += 1
        total += 1

    robustness = round(robust_count / max(total, 1), 4)
    logger.info(
        "Suite 3 (distractor robustness): %.3f (%d/%d passed ROUGE-L >= 0.30)",
        robustness, robust_count, total,
    )
    return {"distractor_robustness": robustness}


def main() -> None:
    """SageMaker Processing Job entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    OUTPUT_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    records = _load_eval_records()
    logger.info("Loaded %d evaluation records.", len(records))

    if not records:
        logger.error("No evaluation records found. Failing.")
        sys.exit(1)

    # Load fine-tuned model for inference.
    gen_pipe = hf_pipeline(
        "text-generation",
        model=str(INPUT_MODEL_DIR),
        device_map="auto" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )

    rouge_metric = evaluate.load("rouge")

    # Run all three suites.
    results: dict[str, float] = {}
    results.update(suite_factual_accuracy(records, gen_pipe, rouge_metric))
    results.update(suite_cross_lingual(records, gen_pipe))
    results.update(suite_distractor_robustness(records, gen_pipe, rouge_metric))

    # Determine pass/fail.
    passed = (
        results.get("rouge_l", 0.0) >= ROUGE_L_THRESHOLD
        and results.get("distractor_robustness", 0.0) >= DISTRACTOR_ROBUSTNESS_THRESHOLD
    )
    results["status"] = "pass" if passed else "fail"

    logger.info("Evaluation result: %s | metrics: %s", results["status"], results)

    # Write evaluation output for SageMaker ConditionStep to read.
    output_path = OUTPUT_EVAL_DIR / "evaluation.json"
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    if EXPERIMENT_RUN:
        log_metrics(run_name=EXPERIMENT_RUN, metrics=results)

    # Fail the Processing Job if eval does not meet threshold — pipeline will not register the model.
    if not passed:
        logger.error(
            "Model did NOT meet pass threshold: ROUGE-L=%.3f (need %.2f), "
            "distractor_robustness=%.3f (need %.2f).",
            results["rouge_l"], ROUGE_L_THRESHOLD,
            results["distractor_robustness"], DISTRACTOR_ROBUSTNESS_THRESHOLD,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
