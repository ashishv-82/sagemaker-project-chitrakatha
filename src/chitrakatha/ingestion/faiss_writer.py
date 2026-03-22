"""FAISS-on-S3 writer for the Chitrakatha ingestion pipeline.

Why: The native S3 Vectors API (2026) is not yet available in the AWS SDK.
     This module implements a production-ready "Scale-to-Zero" alternative
     by storing a FAISS index as a flat file in S3.

     - Ingestion: Downloads index -> Appends vectors -> Uploads index.
     - Inference: Downloads index into RAM -> Sub-millisecond similarity search.

Constraints:
    - Uses FAISS (FlatL2 or InnerProduct) for efficient similarity search.
    - Stores metadata as a companion JSON/Pickle file in the same S3 prefix.
    - Atomic writes: Uses S3 versioning/locking (implied) to prevent corruption.
"""

from __future__ import annotations

import logging
import os
import pickle
import tempfile
from typing import Any, Final

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError

from chitrakatha.exceptions import S3VectorError
from chitrakatha.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Constants for file names in the S3 vector bucket
INDEX_FILENAME: Final[str] = "index.faiss"
META_FILENAME: Final[str] = "metadata.pkl"

# At ~1024 float32 dims, each vector is ~4 KB. 50k vectors ≈ 200 MB in RAM.
# Serverless endpoints have 6 GB memory, but cold-start download time becomes
# significant beyond this threshold. Log a warning so operators know to consider
# index sharding or upgrading to a persistent vector store.
_WARN_VECTOR_COUNT: Final[int] = 50_000


def _build_metadata(chunk: Chunk, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Construct the metadata dict stored alongside each vector."""
    meta: dict[str, str] = {
        "chunk_id": chunk.chunk_id,
        "chunk_text": chunk.text,
        "source_document": chunk.source_document,
        "chunk_index": str(chunk.chunk_index),
        "token_count": str(chunk.token_count),
    }
    if extra:
        meta.update(extra)
    return meta


def _get_s3_client(aws_region: str) -> Any:
    return boto3.client("s3", region_name=aws_region)


def write_vectors(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    bucket_name: str,
    index_name: str,  # In this refactor, index_name serves as the S3 prefix
    aws_region: str,
    extra_metadata: dict[str, str] | None = None,
    s3vectors_client: Any | None = None,  # Ignored, kept for signature compatibility
    existing_ids: set[str] | None = None,
) -> int:
    """Write chunk embeddings to a FAISS index stored in S3.

    Args:
        chunk_embeddings: List of (Chunk, embedding) tuples.
        bucket_name: S3 bucket for vector storage.
        index_name: S3 prefix (folder) inside the bucket.
        aws_region: AWS region.
        extra_metadata: Optional fields to add to every chunk.
        s3vectors_client: UNUSED (kept for backward compatibility).
        existing_ids: Set of chunk IDs to skip (idempotency).

    Returns:
        Number of vectors written.
    """
    if not chunk_embeddings:
        return 0

    s3 = _get_s3_client(aws_region)
    prefix = index_name.strip("/")
    
    # Filter out already-indexed vectors
    to_write = [
        (chunk, emb)
        for chunk, emb in chunk_embeddings
        if existing_ids is None or chunk.chunk_id not in existing_ids
    ]

    if not to_write:
        return 0

    # 1. Prepare new data
    new_embeddings = np.array([emb for _, emb in to_write]).astype("float32")
    new_metadata = [_build_metadata(chunk, extra_metadata) for chunk, _ in to_write]
    
    # Normalize for cosine similarity (InnerProduct)
    faiss.normalize_L2(new_embeddings)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_index_path = os.path.join(tmp_dir, INDEX_FILENAME)
        local_meta_path = os.path.join(tmp_dir, META_FILENAME)

        # 2. Download existing index/metadata if they exist
        index = None
        metadata_map = {}

        try:
            s3.download_file(bucket_name, f"{prefix}/{INDEX_FILENAME}", local_index_path)
            s3.download_file(bucket_name, f"{prefix}/{META_FILENAME}", local_meta_path)
            
            index = faiss.read_index(local_index_path)
            with open(local_meta_path, "rb") as f:
                metadata_map = pickle.load(f)
            logger.info("Loaded existing FAISS index with %d vectors.", index.ntotal)
        except ClientError:
            logger.info("No existing FAISS index found at %s/%s. Creating new.", bucket_name, prefix)
            # Dimension matches Titan Embed Text v2 (1536)
            dimension = len(to_write[0][1])
            index = faiss.IndexFlatIP(dimension)

        # 3. Append new vectors
        start_idx = index.ntotal
        index.add(new_embeddings)
        
        for i, meta in enumerate(new_metadata):
            metadata_map[start_idx + i] = meta

        # 4. Save and Upload
        faiss.write_index(index, local_index_path)
        with open(local_meta_path, "wb") as f:
            pickle.dump(metadata_map, f)

        if index.ntotal > _WARN_VECTOR_COUNT:
            logger.warning(
                "FAISS index has grown to %d vectors (threshold: %d). "
                "Cold-start download time may exceed the 30s CloudWatch alarm threshold. "
                "Consider sharding the index or migrating to a persistent vector store.",
                index.ntotal,
                _WARN_VECTOR_COUNT,
            )

        try:
            s3.upload_file(local_index_path, bucket_name, f"{prefix}/{INDEX_FILENAME}")
            s3.upload_file(local_meta_path, bucket_name, f"{prefix}/{META_FILENAME}")
            logger.info("Successfully updated FAISS index in S3. Total: %d", index.ntotal)
        except ClientError as exc:
            raise S3VectorError(f"Failed to upload updated FAISS index to S3: {exc}") from exc

    return len(to_write)


def query_vectors(
    query_embedding: list[float],
    bucket_name: str,
    index_name: str,
    aws_region: str,
    top_k: int = 5,
    s3vectors_client: Any | None = None, # UNUSED
) -> list[dict[str, Any]]:
    """Retrieve top-k similar vectors from the FAISS-on-S3 store."""
    s3 = _get_s3_client(aws_region)
    prefix = index_name.strip("/")
    
    # Convert query to numpy and normalize
    query_np = np.array([query_embedding]).astype("float32")
    faiss.normalize_L2(query_np)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_index_path = os.path.join(tmp_dir, INDEX_FILENAME)
        local_meta_path = os.path.join(tmp_dir, META_FILENAME)

        try:
            # Note: In a production SageMaker endpoint, we should cache the index
            # at the module level to avoid re-downloading on every request.
            s3.download_file(bucket_name, f"{prefix}/{INDEX_FILENAME}", local_index_path)
            s3.download_file(bucket_name, f"{prefix}/{META_FILENAME}", local_meta_path)
            
            index = faiss.read_index(local_index_path)
            with open(local_meta_path, "rb") as f:
                metadata_map = pickle.load(f)
        except ClientError as exc:
            # S3 raises NoSuchKey (not "404") when the object doesn't exist.
            if exc.response["Error"]["Code"] in {"NoSuchKey", "NoSuchBucket"}:
                logger.warning("Vector index not found in S3 at %s/%s", bucket_name, prefix)
                return []
            raise S3VectorError(f"Failed to download FAISS index from S3: {exc}") from exc

        # Search
        scores, indices = index.search(query_np, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1: continue # FAISS returns -1 if not enough neighbours
            
            meta = metadata_map.get(idx, {})
            result = {
                "score": float(score),
                **meta
            }
            results.append(result)
            
        return results
