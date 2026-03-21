"""S3 Vectors writer for the Chitrakatha ingestion pipeline.

Why: S3 Vectors (2026 native API) is the serverless vector store. This module
     abstracts the boto3 ``s3vectors`` client so all callers (ingest_to_vectors.py
     and serving/inference.py) share a single, tested interface.

     Idempotency is enforced at the vector_id level: if a vector with the same
     ID already exists in the index, the write is skipped. This makes the
     ingestion pipeline safe to re-run without duplicating the index.

Metadata payload per vector:
    ``source_entity``    — Comic character name (e.g. "Nagraj")
    ``publisher``        — Publisher name (e.g. "Raj Comics")
    ``language``         — "en", "hi", or "en-hi"
    ``chunk_text``       — Full text of the chunk (stored for RAG prompt building)
    ``source_document``  — Source filename for lineage
    ``chunk_index``      — Position in source document

Constraints:
    - Raises ``S3VectorError`` on any API failure — no silent swallowing.
    - vector_id is the Chunk's ``chunk_id`` (UUID4) — globally unique.
    - Batch write: up to 100 vectors per API call (S3 Vectors API limit).
"""

from __future__ import annotations

import logging
from typing import Any, Final

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from chitrakatha.exceptions import S3VectorError
from chitrakatha.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# S3 Vectors API max vectors per PutVectors call.
_MAX_WRITE_BATCH: Final[int] = 100


def _build_metadata(chunk: Chunk, extra: dict[str, str] | None = None) -> dict[str, str]:
    """Construct the metadata dict stored alongside each vector.

    All values are strings (S3 Vectors metadata constraint).

    Args:
        chunk: The chunk whose metadata is being built.
        extra: Optional caller-supplied fields (e.g. source_entity, publisher).

    Returns:
        Metadata dict with all required fields populated.
    """
    meta: dict[str, str] = {
        "chunk_text": chunk.text,
        "source_document": chunk.source_document,
        "chunk_index": str(chunk.chunk_index),
        "token_count": str(chunk.token_count),
    }
    if extra:
        meta.update(extra)
    return meta


def write_vectors(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    bucket_name: str,
    index_name: str,
    aws_region: str,
    extra_metadata: dict[str, str] | None = None,
    s3vectors_client: Any | None = None,
    existing_ids: set[str] | None = None,
) -> int:
    """Write chunk embeddings to the S3 Vectors index in batches.

    Args:
        chunk_embeddings: List of ``(Chunk, embedding)`` tuples from the embedder.
        bucket_name: Name of the S3 Vectors bucket.
        index_name: Name of the vector index within the bucket.
        aws_region: AWS region for the S3 Vectors endpoint.
        extra_metadata: Optional fields added to every vector's metadata
            (e.g. ``{"source_entity": "Nagraj", "publisher": "Raj Comics"}``).
        s3vectors_client: Optional pre-built ``s3vectors`` boto3 client
            (injected for testing).
        existing_ids: Set of vector IDs already present in the index.
            Used for idempotency — pass ``None`` to skip the pre-check.

    Returns:
        Number of vectors actually written (skipped duplicates not counted).

    Raises:
        S3VectorError: On any API failure during batch write.
        ValueError: If ``chunk_embeddings`` is empty.
    """
    if not chunk_embeddings:
        raise ValueError("write_vectors received an empty chunk_embeddings list.")

    client = s3vectors_client or boto3.client(
        "s3vectors", region_name=aws_region
    )

    # Filter out already-indexed vectors for idempotency.
    to_write = [
        (chunk, emb)
        for chunk, emb in chunk_embeddings
        if existing_ids is None or chunk.chunk_id not in existing_ids
    ]

    skipped = len(chunk_embeddings) - len(to_write)
    if skipped:
        logger.info("Skipping %d already-indexed vector(s).", skipped)

    written_count = 0

    for batch_start in range(0, len(to_write), _MAX_WRITE_BATCH):
        batch = to_write[batch_start : batch_start + _MAX_WRITE_BATCH]

        vectors_payload = [
            {
                "Key": chunk.chunk_id,
                "Data": {"Float32": emb},
                "Metadata": _build_metadata(chunk, extra_metadata),
            }
            for chunk, emb in batch
        ]

        try:
            client.put_vectors(
                VectorBucketName=bucket_name,
                IndexName=index_name,
                Vectors=vectors_payload,
            )
            written_count += len(batch)
            logger.info(
                "Wrote batch of %d vectors to index '%s' (total so far: %d).",
                len(batch), index_name, written_count,
            )
        except (BotoCoreError, ClientError) as exc:
            raise S3VectorError(
                f"Failed to write vector batch (items {batch_start}–"
                f"{batch_start + len(batch)}) to index '{index_name}': {exc}"
            ) from exc

    return written_count


def query_vectors(
    query_embedding: list[float],
    bucket_name: str,
    index_name: str,
    aws_region: str,
    top_k: int = 5,
    s3vectors_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Retrieve the top-k most similar vectors to ``query_embedding``.

    Called by ``serving/inference.py`` at query time.

    Args:
        query_embedding: 1536-dim query vector from ``embedder.embed_query()``.
        bucket_name: Name of the S3 Vectors bucket.
        index_name: Name of the vector index.
        aws_region: AWS region.
        top_k: Number of nearest neighbours to return. Default 5.
        s3vectors_client: Optional pre-built client (for testing).

    Returns:
        List of result dicts containing ``chunk_text``, ``score``, and all
        stored metadata fields.

    Raises:
        S3VectorError: On API failure.
    """
    client = s3vectors_client or boto3.client(
        "s3vectors", region_name=aws_region
    )

    try:
        response = client.query_vectors(
            VectorBucketName=bucket_name,
            IndexName=index_name,
            QueryVector={"Float32": query_embedding},
            TopK=top_k,
            ReturnMetadata=True,
        )
    except (BotoCoreError, ClientError) as exc:
        raise S3VectorError(
            f"Failed to query vector index '{index_name}': {exc}"
        ) from exc

    results: list[dict[str, Any]] = []
    for match in response.get("Vectors", []):
        result = {"score": match.get("Score", 0.0)}
        result.update(match.get("Metadata", {}))
        results.append(result)

    return results
