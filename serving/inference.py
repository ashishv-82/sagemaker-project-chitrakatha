"""SageMaker Endpoint Inference Entry Point.

Why: This script runs inside the HuggingFace PyTorch Inference container
     when the Serverless Endpoint is invoked. It handles the complete RAG
     (Retrieval-Augmented Generation) flow during a live user request.

     Transition (v2): Switched from simulated s3vectors API to real FAISS-on-S3.
     Index is cached at the module level to minimize cold-start latency.

RAG Request Flow (``predict_fn``):
    1. Receive user query (JSON).
    2. Embed the query using Bedrock Titan Embed v2.
    3. Retrieve top-k chunks from FAISS index (downloaded from S3 if not cached).
    4. Construct the RAFT-style prompt (query + retrieved chunks).
    5. Call the fine-tuned Llama 3.1 8B model to generate the response.
    6. Return response + listed sources back to the client.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import tempfile
from typing import Any

import boto3
import faiss
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

logger = logging.getLogger(__name__)

# Constants for file names in the S3 vector bucket
INDEX_FILENAME = "index.faiss"
META_FILENAME = "metadata.pkl"

# Passed down by deploy_endpoint.py
S3_VECTORS_BUCKET = os.environ.get("S3_VECTORS_BUCKET")
S3_FAISS_INDEX_PREFIX = os.environ.get("S3_FAISS_INDEX_PREFIX")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")

# Module-level cache for the FAISS index and metadata
_INDEX_CACHE: dict[str, Any] = {
    "index": None,
    "metadata": None,
}


def model_fn(model_dir: str) -> Any:
    """Load the trained model and tokenizer from model_dir during container boot."""
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


def _load_index() -> tuple[Any, dict[int, Any]]:
    """Download and load the FAISS index from S3 with caching."""
    if _INDEX_CACHE["index"] is not None:
        return _INDEX_CACHE["index"], _INDEX_CACHE["metadata"]

    if not S3_VECTORS_BUCKET or not S3_FAISS_INDEX_PREFIX:
        raise ValueError("S3_VECTORS_BUCKET or S3_FAISS_INDEX_PREFIX not configured.")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    prefix = S3_FAISS_INDEX_PREFIX.strip("/")

    logger.info("Downloading FAISS index from s3://%s/%s", S3_VECTORS_BUCKET, prefix)
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        local_index_path = os.path.join(tmp_dir, INDEX_FILENAME)
        local_meta_path = os.path.join(tmp_dir, META_FILENAME)

        try:
            s3.download_file(S3_VECTORS_BUCKET, f"{prefix}/{INDEX_FILENAME}", local_index_path)
            s3.download_file(S3_VECTORS_BUCKET, f"{prefix}/{META_FILENAME}", local_meta_path)
            
            _INDEX_CACHE["index"] = faiss.read_index(local_index_path)
            with open(local_meta_path, "rb") as f:
                _INDEX_CACHE["metadata"] = pickle.load(f)
            
            logger.info("Index loaded successfully with %d vectors.", _INDEX_CACHE["index"].ntotal)
            return _INDEX_CACHE["index"], _INDEX_CACHE["metadata"]
        except Exception as exc:
            logger.error("Failed to load FAISS index from S3: %s", exc)
            return None, {}


def _retrieve_vectors(embedding: list[float], top_k: int = 5) -> list[dict[str, Any]]:
    """Retrieve similar vectors using the local FAISS index."""
    index, metadata_map = _load_index()
    if index is None:
        return []

    # FAISS search
    query_np = np.array([embedding]).astype("float32")
    faiss.normalize_L2(query_np)
    
    scores, indices = index.search(query_np, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1: continue
        
        meta = metadata_map.get(idx, {})
        results.append({
            "score": float(score),
            "text": meta.get("chunk_text", ""),
            "source_document": meta.get("source_document", "Unknown"),
            "publisher": meta.get("publisher", "Unknown"),
        })
    return results


def _build_raft_prompt(query: str, retrieved_docs: list[dict[str, Any]]) -> str:
    """Construct the RAFT prompt format used during training."""
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

    logger.info("Received query: %s", query)

    # 1. Embed Query
    query_embedding = _embed_query(query)

    # 2. Retrieve Top-5
    retrieved = _retrieve_vectors(query_embedding, top_k=5)
    if not retrieved:
        return {
            "answer": "I don't have enough context to answer that from the comic archives.",
            "sources": [],
        }

    # 3. Format Prompt
    prompt = _build_raft_prompt(query, retrieved)

    # 4. Generate
    outputs = model_pipeline(
        prompt,
        max_new_tokens=256,
        do_sample=False,
        temperature=None,
        top_p=None,
        return_full_text=False,
    )
    generated_text = outputs[0]["generated_text"].strip()

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
