"""Flow A orchestration: Silver corpus → chunk → embed → S3 Vectors index.

Why: This script is the top-level orchestrator for the RAG knowledge base
     ingestion path. It reads clean JSONL from S3 Silver /corpus/, chunks
     each document, embeds in batches via Bedrock Titan Embed v2, and writes
     to the S3 Vectors index. It is designed to be run as a SageMaker
     Processing Job (see pipeline/steps/embed_and_index.py).

Idempotency:
    Before writing, the script fetches all existing vector IDs from the index
    and passes them to ``vector_writer.write_vectors()`` to skip re-embedding
    unchanged documents. This makes the pipeline safe to re-trigger without
    bloating the index.

Input format (S3 Silver /corpus/ objects):
    UTF-8 JSON Lines, one document per line:
    ``{"text": "...", "source_document": "nagraj_wiki.txt", "language": "en"}``

Constraints:
    - All config sourced from ``Settings`` (environment variables from
      Terraform outputs) — no hardcoded bucket names.
    - Never crashes silently; any document-level error is logged and counted.
    - Exits with code 1 if > 0 documents fail (so the SageMaker step fails).
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from chitrakatha.config import Settings
from chitrakatha.exceptions import BedrockEmbeddingError, DataIngestionError, S3VectorError
from chitrakatha.ingestion.chunker import chunk_text
from chitrakatha.ingestion.embedder import embed_chunks
from chitrakatha.ingestion.vector_writer import query_vectors, write_vectors

logger = logging.getLogger(__name__)

# S3 Silver prefix containing clean corpus JSONL files (Flow A input).
_SILVER_CORPUS_PREFIX: str = "corpus/"


def _list_silver_objects(
    s3_client: boto3.client,
    bucket: str,
    prefix: str,
) -> list[str]:
    """List all object keys under ``prefix`` in the Silver bucket.

    Args:
        s3_client: Boto3 S3 client.
        bucket: Silver bucket name.
        prefix: Key prefix to list (e.g. ``"corpus/"``).

    Returns:
        Sorted list of S3 object keys.
    """
    keys: list[str] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return sorted(keys)


def _load_existing_vector_ids(
    settings: Settings,
    s3vectors_client: boto3.client,
) -> set[str]:
    """Fetch all existing vector IDs from the index for idempotency checking.

    Args:
        settings: Runtime settings.
        s3vectors_client: Boto3 s3vectors client.

    Returns:
        Set of existing vector ID strings (chunk_ids).
    """
    existing: set[str] = []
    try:
        paginator = s3vectors_client.get_paginator("list_vectors")
        for page in paginator.paginate(
            VectorBucketName=settings.s3_vectors_bucket,
            IndexName=settings.s3_vector_index_name,
        ):
            for vec in page.get("Vectors", []):
                existing.append(vec["Key"])
    except (BotoCoreError, ClientError) as exc:
        logger.warning(
            "Could not list existing vectors (may be first run): %s. "
            "Proceeding without idempotency check.",
            exc,
        )
        return set()
    return set(existing)


def run(settings: Settings) -> int:
    """Execute the full Flow A ingestion pipeline.

    Reads every JSONL file from S3 Silver /corpus/, chunks, embeds, and
    writes to the S3 Vectors index. Counts and reports errors.

    Args:
        settings: Pydantic settings from environment (Terraform outputs).

    Returns:
        Number of documents that failed processing.
    """
    s3_client = boto3.client("s3", region_name=settings.aws_region)
    s3vectors_client = boto3.client("s3vectors", region_name=settings.aws_region)

    # Pre-load existing IDs for idempotency.
    existing_ids = _load_existing_vector_ids(settings, s3vectors_client)
    logger.info("Found %d existing vectors in index.", len(existing_ids))

    keys = _list_silver_objects(s3_client, settings.s3_silver_bucket, _SILVER_CORPUS_PREFIX)
    logger.info("Found %d JSONL objects to process in Silver /corpus/.", len(keys))

    errors = 0
    total_written = 0

    for key in keys:
        try:
            response = s3_client.get_object(
                Bucket=settings.s3_silver_bucket,
                Key=key,
            )
            raw = response["Body"].read().decode("utf-8")
        except (BotoCoreError, ClientError) as exc:
            logger.error("Failed to read s3://%s/%s: %s", settings.s3_silver_bucket, key, exc)
            errors += 1
            continue

        for line_no, line in enumerate(raw.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
            except json.JSONDecodeError as exc:
                logger.error("Bad JSON at %s line %d: %s", key, line_no, exc)
                errors += 1
                continue

            text: str = doc.get("text", "")
            source_document: str = doc.get("source_document", key)
            language: str = doc.get("language", "en")

            try:
                chunks = chunk_text(text, source_document=source_document)
                chunk_embeddings = embed_chunks(
                    chunks,
                    aws_region=settings.aws_region,
                )
                written = write_vectors(
                    chunk_embeddings,
                    bucket_name=settings.s3_vectors_bucket,
                    index_name=settings.s3_vector_index_name,
                    aws_region=settings.aws_region,
                    extra_metadata={"language": language, "source_document": source_document},
                    existing_ids=existing_ids,
                )
                total_written += written
                # Update local cache so subsequent documents don't re-check already-written IDs.
                for chunk, _ in chunk_embeddings:
                    existing_ids.add(chunk.chunk_id)

            except (DataIngestionError, BedrockEmbeddingError, S3VectorError) as exc:
                logger.error("Failed to ingest document '%s': %s", source_document, exc)
                errors += 1

    logger.info(
        "Flow A complete. Written: %d new vector(s). Errors: %d document(s).",
        total_written, errors,
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
