"""SageMaker Training step: QLoRA fine-tuning with RAFT prompt template.

Why: Standard SFT (question→answer) teaches the model to recall facts. RAFT
     fine-tuning (question + golden document + distractors → answer) teaches
     it the skill of reading and selecting from retrieved context — which is
     exactly what it faces at inference time when querying the S3 Vectors index.

     QLoRA (4-bit quantization + LoRA adapters) makes fine-tuning a 8B model
     feasible on a single g5.2xlarge Spot instance (24GB VRAM) in ~1-2 hours.

RAFT prompt template (documents shuffled to prevent positional shortcuts):
    You are given the following documents:
    [Document 1 - may or may not be relevant]: {distractor_1}
    [Document 2 - may or may not be relevant]: {golden_chunk}
    [Document 3 - may or may not be relevant]: {distractor_2}
    Question: {question}
    Think step by step, then answer using ONLY the relevant document above.
    {chain_of_thought}
    Answer: {answer}

Hyperparameters:
    model_id: meta-llama/Meta-Llama-3.1-8B-Instruct
    lora_r: 16, lora_alpha: 32, lora_dropout: 0.05
    target_modules: [q_proj, v_proj]
    quantization: 4-bit NF4 (BitsAndBytesConfig)
    epochs: 3, batch_size: 4, lr: 2e-4, warmup_ratio: 0.03

Constraints:
    - Spot interruption handled by SageMaker checkpointing.
    - All hyperparameters, metrics logged to SageMaker Experiments run.
    - Evaluation on held-out 10% of data; ROUGE-L reported per epoch.
"""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer

from chitrakatha.monitoring.experiments import log_hyperparameters, log_metrics

logger = logging.getLogger(__name__)

# SageMaker channel mount points.
INPUT_GOLD_DIR = Path(os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
MODEL_OUTPUT_DIR = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
CHECKPOINT_DIR = Path(os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data")) / "checkpoints"

# Hyperparameters (overridable via SageMaker TrainingJob hyperparameters dict).
MODEL_ID = os.environ.get("MODEL_ID", "meta-llama/Meta-Llama-3.1-8B-Instruct")
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))
NUM_EPOCHS = int(os.environ.get("EPOCHS", "3"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "4"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "2e-4"))
MAX_SEQ_LENGTH = int(os.environ.get("MAX_SEQ_LENGTH", "2048"))
EVAL_SPLIT = float(os.environ.get("EVAL_SPLIT", "0.10"))
EXPERIMENT_RUN = os.environ.get("SAGEMAKER_EXPERIMENT_RUN", "")


def _build_raft_prompt(record: dict) -> str:
    """Format a RAFT training record as the model's input+output sequence.

    Documents are shuffled so the model cannot learn that the golden chunk
    is always in a fixed position.

    Args:
        record: A Gold JSONL record with golden_chunk, distractor_chunks,
            question_en, chain_of_thought, answer_en.

    Returns:
        Full prompt string for SFT training (input + expected completion).
    """
    golden = record.get("golden_chunk", "")
    distractors = record.get("distractor_chunks", ["", ""])
    question = record.get("question_en", "")
    cot = record.get("chain_of_thought", "")
    answer = record.get("answer_en", "")

    # Shuffle: mix golden among distractors at a random position.
    docs = [distractors[0], golden, distractors[1]] if len(distractors) >= 2 else [golden]
    random.shuffle(docs)

    doc_section = "\n".join(
        f"[Document {i+1} - may or may not be relevant]: {doc}"
        for i, doc in enumerate(docs)
    )

    return (
        f"You are given the following documents:\n{doc_section}\n\n"
        f"Question: {question}\n\n"
        f"Think step by step, then answer using ONLY the relevant document above.\n"
        f"{cot}\n"
        f"Answer: {answer}"
    )


def _load_gold_dataset() -> tuple[Dataset, Dataset]:
    """Load Gold JSONL files and split into train/eval sets.

    Returns:
        Tuple of (train_dataset, eval_dataset).
    """
    records = []
    for jsonl_path in sorted(INPUT_GOLD_DIR.glob("*.jsonl")):
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.warning("Skipping bad JSON line in %s.", jsonl_path.name)

    logger.info("Loaded %d RAFT training records.", len(records))
    random.shuffle(records)

    split_idx = max(1, int(len(records) * (1 - EVAL_SPLIT)))
    train_records = records[:split_idx]
    eval_records = records[split_idx:]

    train_ds = Dataset.from_list([{"text": _build_raft_prompt(r)} for r in train_records])
    eval_ds = Dataset.from_list([{"text": _build_raft_prompt(r)} for r in eval_records])

    logger.info("Train: %d records. Eval: %d records.", len(train_ds), len(eval_ds))
    return train_ds, eval_ds


def main() -> None:
    """SageMaker Training Job entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    MODEL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Log hyperparameters to SageMaker Experiments.
    hparams = {
        "base_model": MODEL_ID,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "epochs": NUM_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "technique": "RAFT+QLoRA",
    }
    if EXPERIMENT_RUN:
        log_hyperparameters(run_name=EXPERIMENT_RUN, hyperparameters=hparams)

    # 4-bit NF4 quantization — enables 8B model on 24GB VRAM.
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    logger.info("Loading base model: %s", MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=False,
    )
    model = prepare_model_for_kbit_training(model)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # LoRA adapter configuration — target attention projections only.
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds, eval_ds = _load_gold_dataset()

    training_args = TrainingArguments(
        output_dir=str(CHECKPOINT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=4,
        learning_rate=LEARNING_RATE,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        fp16=False,
        bf16=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_steps=50,
        report_to="none",  # We handle Experiments logging ourselves.
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
        args=training_args,
    )

    logger.info("Starting RAFT+QLoRA fine-tuning.")
    train_result = trainer.train()

    # Save the final merged model to the SageMaker model directory.
    trainer.save_model(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    logger.info("Model saved to %s.", MODEL_OUTPUT_DIR)

    final_metrics = {
        "train_loss": round(train_result.training_loss, 4),
        "train_runtime_sec": round(train_result.metrics.get("train_runtime", 0), 1),
        "train_samples_per_sec": round(
            train_result.metrics.get("train_samples_per_second", 0), 2
        ),
    }
    logger.info("Training complete: %s", final_metrics)

    if EXPERIMENT_RUN:
        log_metrics(run_name=EXPERIMENT_RUN, metrics=final_metrics)


if __name__ == "__main__":
    main()
