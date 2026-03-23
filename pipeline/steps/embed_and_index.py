"""SageMaker Processing step: Flow A — Silver /corpus/ → pgvector RDS index.

Why: Reads the corpus JSONL produced by preprocessing.py, embeds each chunk
     via Bedrock Titan Embed v2, and upserts vectors into the pgvector RDS table.
     Running as a SageMaker Processing Job ensures reproducible execution with
     automatic lineage tracking — vector counts are logged to SageMaker Experiments.

Input channel (SageMaker):
    /opt/ml/processing/input/corpus — corpus.jsonl from preprocessing step

Idempotency: pgvector uses ON CONFLICT (source_document, chunk_index) DO NOTHING,
             so re-running this step never duplicates rows.

Constraints:
    - Exits with code 1 on any embedding or write failure.
    - Logs ``vector_count_written`` metric to SageMaker Experiments run.
"""

from __future__ import annotations

import subprocess
import sys

# Install runtime deps before any chitrakatha imports.
subprocess.check_call([
    sys.executable, "-m", "pip", "install",
    "pydantic>=2.5.0", "pydantic-settings>=2.2.0",
    "psycopg2-binary>=2.9.0", "pgvector>=0.3.0",
    "numpy>=1.26.0", "pytz",
    "--quiet",
])

import json
import logging
import os
from pathlib import Path

from chitrakatha.config import Settings
from chitrakatha.exceptions import BedrockEmbeddingError, DataIngestionError, PgVectorError
from chitrakatha.ingestion.chunker import chunk_text
from chitrakatha.ingestion.embedder import embed_chunks
from chitrakatha.ingestion.pgvector_writer import write_vectors
from chitrakatha.monitoring.experiments import log_metrics

logger = logging.getLogger(__name__)

INPUT_CORPUS_PATH = Path("/opt/ml/processing/input/corpus/corpus.jsonl")


def run(settings: Settings, experiment_run_name: str | None = None) -> int:
    """Embed Silver corpus chunks and write to pgvector RDS.

    Args:
        settings: Runtime settings from environment.
        experiment_run_name: SageMaker Experiments run name for metric logging.

    Returns:
        Error count (0 = success).
    """
    if not INPUT_CORPUS_PATH.exists():
        logger.error("Corpus input file not found: %s", INPUT_CORPUS_PATH)
        return 1

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
                db_secret_arn=settings.db_secret_arn,
                aws_region=settings.aws_region,
            )
            total_written += written
            logger.info(
                "Document '%s' (lang=%s): %d chunk(s), %d newly written.",
                source_document, language, len(chunks), written,
            )

        except (DataIngestionError, BedrockEmbeddingError, PgVectorError) as exc:
            logger.error("Failed document '%s': %s", source_document, exc)
            errors += 1

    logger.info(
        "embed_and_index complete. Total new vectors: %d. Errors: %d.",
        total_written, errors,
    )

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
