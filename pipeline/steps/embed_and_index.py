"""SageMaker Processing step: Flow A — Silver /corpus/ → S3 Vectors index.

Why: This step runs after preprocessing.py and is responsible for building
     the RAG knowledge base. It reads the corpus JSONL produced by
     preprocessing.py, chunks and embeds each document, and writes vectors
     to the S3 Vectors index.

     Running as a SageMaker Processing Job ensures idempotent, reproducible
     execution with automatic lineage tracking — the vector count is logged
     to SageMaker Experiments so we can track knowledge base growth over runs.

Input channel (SageMaker):
    /opt/ml/processing/input/corpus — corpus.jsonl from preprocessing step

Constraints:
    - Idempotent: skips re-embedding of already-indexed chunks.
    - Logs ``vector_count_written`` metric to SageMaker Experiments run.
    - Exits with code 1 on any embedding or write failure.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import boto3

from chitrakatha.config import Settings
from chitrakatha.exceptions import BedrockEmbeddingError, DataIngestionError, S3VectorError
from chitrakatha.ingestion.chunker import chunk_text
from chitrakatha.ingestion.embedder import embed_chunks
from chitrakatha.ingestion.vector_writer import write_vectors
from chitrakatha.monitoring.experiments import log_metrics

logger = logging.getLogger(__name__)

# SageMaker Processing Job channel mount point for corpus input.
INPUT_CORPUS_PATH = Path("/opt/ml/processing/input/corpus/corpus.jsonl")


def _load_existing_vector_ids(settings: Settings) -> set[str]:
    """Pre-load existing vector IDs for idempotency. Returns empty set on failure."""
    s3vectors_client = boto3.client("s3vectors", region_name=settings.aws_region)
    existing: set[str] = set()
    try:
        paginator = s3vectors_client.get_paginator("list_vectors")
        for page in paginator.paginate(
            VectorBucketName=settings.s3_vectors_bucket,
            IndexName=settings.s3_vector_index_name,
        ):
            for vec in page.get("Vectors", []):
                existing.add(vec["Key"])
    except Exception as exc:
        logger.warning("Could not list existing vectors (first run?): %s", exc)
    return existing


def run(settings: Settings, experiment_run_name: str | None = None) -> int:
    """Embed and index Silver corpus into S3 Vectors.

    Args:
        settings: Runtime settings from environment.
        experiment_run_name: SageMaker Experiments run name for metric logging.

    Returns:
        Error count (0 = success).
    """
    if not INPUT_CORPUS_PATH.exists():
        logger.error("Corpus input file not found: %s", INPUT_CORPUS_PATH)
        return 1

    existing_ids = _load_existing_vector_ids(settings)
    logger.info("Pre-loaded %d existing vector ID(s) for idempotency.", len(existing_ids))

    raw_lines = INPUT_CORPUS_PATH.read_text(encoding="utf-8").splitlines()
    errors = 0
    total_written = 0

    for line_no, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            continue
        try:
            doc = json.loads(line)
        except json.JSONDecodeError as exc:
            logger.error("Bad JSON at line %d: %s", line_no, exc)
            errors += 1
            continue

        text = doc.get("text", "")
        source_document = doc.get("source_document", f"line_{line_no}")
        language = doc.get("language", "en")

        try:
            chunks = chunk_text(text, source_document=source_document)
            chunk_embeddings = embed_chunks(chunks, aws_region=settings.aws_region)
            written = write_vectors(
                chunk_embeddings,
                bucket_name=settings.s3_vectors_bucket,
                index_name=settings.s3_vector_index_name,
                aws_region=settings.aws_region,
                extra_metadata={
                    "language": language,
                    "source_document": source_document,
                },
                existing_ids=existing_ids,
            )
            total_written += written
            for chunk, _ in chunk_embeddings:
                existing_ids.add(chunk.chunk_id)

        except (DataIngestionError, BedrockEmbeddingError, S3VectorError) as exc:
            logger.error("Failed document '%s': %s", source_document, exc)
            errors += 1

    logger.info(
        "embed_and_index complete. New vectors written: %d. Errors: %d.",
        total_written, errors,
    )

    # Log vector count to SageMaker Experiments for tracking knowledge base growth.
    if experiment_run_name:
        log_metrics(
            run_name=experiment_run_name,
            metrics={"vector_count_written": total_written, "embedding_errors": errors},
        )

    return errors


def main() -> None:
    """SageMaker Processing Job entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = Settings()
    experiment_run_name = os.environ.get("SAGEMAKER_EXPERIMENT_RUN")
    error_count = run(settings, experiment_run_name)
    sys.exit(1 if error_count > 0 else 0)


if __name__ == "__main__":
    main()
