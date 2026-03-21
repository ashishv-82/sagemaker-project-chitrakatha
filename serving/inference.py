"""SageMaker Endpoint Inference Entry Point.

Why: This script runs inside the HuggingFace PyTorch Inference container
     when the Serverless Endpoint is invoked. It handles the complete RAG
     (Retrieval-Augmented Generation) flow during a live user request.

RAG Request Flow (``predict_fn``):
    1. Receive user query (JSON).
    2. Embed the query using Bedrock Titan Embed v2.
    3. Retrieve top-5 most relevant chunks from S3 Vectors index.
    4. Construct the RAFT-style prompt (query + retrieved chunks).
    5. Call the fine-tuned Llama 3.1 8B model to generate the response.
    6. Return response + listed sources back to the client.

Features:
    - Cross-lingual generation automatically handled by the fine-tuned model
      (if query indicates Hindi, answer comes back in Hindi).
    - Checks env vars injected by ``deploy_endpoint.py`` for S3 Vectors config.
    - Gracefully handles empty retrieval contexts.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import boto3
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Passed down by deploy_endpoint.py
S3_VECTORS_BUCKET = os.environ.get("S3_VECTORS_BUCKET")
S3_VECTOR_INDEX_NAME = os.environ.get("S3_VECTOR_INDEX_NAME")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")


def model_fn(model_dir: str) -> Any:
    """Load the trained model and tokenizer from model_dir during container boot.

    Args:
        model_dir: Directory containing model artifacts (downloaded by SageMaker).

    Returns:
        A loaded HuggingFace ``pipeline`` object ready for text generation.
    """
    logger.info("Loading model from %s", model_dir)
    device_map = "auto" if torch.cuda.is_available() else "cpu"

    # Load tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)

    # Load model (QLoRA adapters are already merged from train.py).
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )

    logger.info("Model loaded successfully on %s", device_map)

    # We use a pipeline for simplicity in generation.
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
    )


def _embed_query(query: str) -> list[float]:
    """Embed the user's query using Bedrock Titan v2."""
    bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
    body = json.dumps({"inputText": query})
    try:
        resp = bedrock.invoke_model(
            modelId="amazon.titan-embed-text-v2:0",
            contentType="application/json",
            accept="application/json",
            body=body,
        )
        data = json.loads(resp["body"].read())
        return data["embedding"]
    except Exception as exc:
        logger.error("Error embedding query via Bedrock: %s", exc)
        raise ValueError(f"Embedding failed: {exc}") from exc


def _retrieve_vectors(embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Query S3 Vectors to find closest matching text chunks."""
    if not S3_VECTORS_BUCKET or not S3_VECTOR_INDEX_NAME:
        raise ValueError("S3 Vectors environment variables not set.")

    s3_vectors = boto3.client("s3vectors", region_name=AWS_REGION)
    try:
        resp = s3_vectors.query_vectors(
            VectorBucketName=S3_VECTORS_BUCKET,
            IndexName=S3_VECTOR_INDEX_NAME,
            Vector=embedding,
            TopK=top_k,
        )
    except Exception as exc:
        logger.error("Error querying S3 Vectors: %s", exc)
        raise ValueError(f"Vector search failed: {exc}") from exc

    results = []
    for match in resp.get("Matches", []):
        try:
            # Metadata is stored as JSON in the S3 Vectors index.
            metadata = json.loads(match.get("Metadata", "{}"))
            results.append({
                "score": match.get("Score", 0.0),
                "text": metadata.get("chunk_text", ""),
                "source_document": metadata.get("source_document", "Unknown"),
                "publisher": metadata.get("publisher", "Unknown"),
            })
        except json.JSONDecodeError:
            continue

    return results


def _build_raft_prompt(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """Construct the RAFT prompt format used during training.

    [Document N]: {text}
    ...
    Question: {query}
    Think step by step, then answer using ONLY the relevant document above.
    Answer:
    """
    doc_blocks = []
    for i, doc in enumerate(retrieved_docs):
        text = doc["text"]
        doc_blocks.append(f"[Document {i+1}]: {text}")

    context = "\n\n".join(doc_blocks)
    return (
        f"You are given the following documents:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Think step by step, then answer using ONLY the relevant document above.\nAnswer:"
    )


def predict_fn(data: dict | str, model_pipeline: Any) -> list[str] | dict:
    """Handle a single inference request.

    Args:
        data: Payload from SageMaker invoke_endpoint (usually dict).
        model_pipeline: The object returned by ``model_fn``.

    Returns:
        JSON-serializable response dict or string list.
    """
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

    logger.info("Received query: %s", query)

    # Step 1: Embed Query
    query_embedding = _embed_query(query)

    # Step 2: Retrieve Top-5
    retrieved = _retrieve_vectors(query_embedding, top_k=5)
    if not retrieved:
        return {
            "answer": "I don't have enough context to answer that from the comic archives.",
            "sources": [],
        }

    # Step 3: Format Prompt
    prompt = _build_raft_prompt(query, retrieved)

    # Step 4: Generate (Serverless inference caps our tokens; use conservatively).
    outputs = model_pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )
    generated_text = outputs[0]["generated_text"].strip()

    # Log the chain-of-thought but don't return it to end user directly,
    # or just return everything after 'Answer:' if using that format strictly.
    answer_idx = generated_text.rfind("Answer:")
    if answer_idx != -1:
        answer = generated_text[answer_idx + len("Answer:") :].strip()
    else:
        answer = generated_text

    sources = list({doc["source_document"] for doc in retrieved})

    return {
        "answer": answer,
        "sources": sources,
    }
