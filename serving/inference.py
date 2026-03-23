"""SageMaker Endpoint Inference Entry Point — Qwen2.5-3B-Instruct (benchmarking).

Why: This script runs inside the PyTorch Inference container for the fine-tuned
     Qwen2.5-3B model. It is the benchmarking path only — the production serving
     path is Lambda + Bedrock Qwen3 Next 80B A3B.

     Benchmarking flow: embed query (Bedrock Titan Embed v2) → retrieve top-5
     chunks from pgvector RDS → generate answer with the fine-tuned Qwen2.5-3B.

     This lets us compare: fine-tuned 3B (domain specialist) vs Qwen3 Next 80B A3B
     (general knowledge + instruction following) on Indian comic history Q&A.

Env vars (injected by deploy_endpoint.py):
    DB_SECRET_ARN          — Secrets Manager ARN for RDS credentials
    BEDROCK_EMBED_MODEL_ID — Titan Embed v2 model ID
    AWS_REGION             — AWS region
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
import psycopg2
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

logger = logging.getLogger(__name__)

DB_SECRET_ARN = os.environ.get("DB_SECRET_ARN", "")
EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
_TOP_K = 5


def model_fn(model_dir: str) -> Any:
    """Load the fine-tuned Qwen2.5-3B model and tokenizer during container boot.

    Uses 4-bit NF4 quantization on GPU (~4 GB VRAM) or bfloat16 on CPU.
    """
    logger.info("Loading fine-tuned Qwen2.5-3B from %s", model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    if torch.cuda.is_available():
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="auto",
            quantization_config=bnb_config,
        )
        logger.info("Qwen2.5-3B loaded with 4-bit NF4 on GPU.")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
        logger.warning("No GPU — running Qwen2.5-3B in bfloat16 on CPU.")

    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto" if torch.cuda.is_available() else "cpu",
    )


def _get_db_creds() -> dict:
    sm = boto3.client("secretsmanager", region_name=AWS_REGION)
    resp = sm.get_secret_value(SecretId=DB_SECRET_ARN)
    return json.loads(resp["SecretString"])  # type: ignore[no-any-return]


def _embed_query(query: str) -> list[float]:
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": query}),
    )
    return json.loads(resp["body"].read())["embedding"]  # type: ignore[no-any-return]


def _retrieve(embedding: list[float]) -> list[dict[str, str]]:
    creds = _get_db_creds()
    conn = psycopg2.connect(
        host=creds["host"],
        port=int(creds["port"]),
        dbname=creds["dbname"],
        user=creds["username"],
        password=creds["password"],
        connect_timeout=5,
        sslmode="require",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text, source_document
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (str(embedding), _TOP_K),
            )
            rows = cur.fetchall()
        return [{"text": r[0], "source_document": r[1]} for r in rows]
    finally:
        conn.close()


def _build_prompt(query: str, chunks: list[dict[str, str]]) -> str:
    doc_blocks = "\n\n".join(
        f"[Document {i + 1}]: {c['text']}"
        for i, c in enumerate(chunks)
    )
    return (
        f"You are given the following documents:\n{doc_blocks}\n\n"
        f"Question: {query}\n\n"
        "Think step by step, then answer using ONLY the relevant document above.\nAnswer:"
    )


def predict_fn(data: dict | str, model_pipeline: Any) -> dict:
    """Handle a single inference request."""
    if isinstance(data, str):
        try:
            payload = json.loads(data)
        except json.JSONDecodeError:
            payload = {"query": data}
    else:
        payload = data

    query = payload.get("query", "").strip()
    if not query:
        return {"error": "No 'query' provided in payload."}

    embedding = _embed_query(query)
    chunks = _retrieve(embedding)

    if not chunks:
        return {
            "answer": "I don't have enough context to answer that from the comic archives.",
            "sources": [],
        }

    prompt = _build_prompt(query, chunks)
    outputs = model_pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )
    answer = outputs[0]["generated_text"].strip()
    sources = sorted({c["source_document"] for c in chunks})

    return {"answer": answer, "sources": sources}
