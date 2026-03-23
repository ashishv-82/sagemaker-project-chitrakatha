"""PostgreSQL pgvector writer for the Chitrakatha ingestion pipeline.

Why: Replaces FAISS-on-S3 with a managed PostgreSQL/pgvector store in RDS.
     pgvector supports HNSW indexing and cosine similarity queries without
     requiring S3 round-trips on every inference call.

Idempotency: INSERT ... ON CONFLICT (source_document, chunk_index) DO NOTHING
             ensures re-running the pipeline never duplicates rows.

Schema init: CREATE EXTENSION / TABLE / INDEX are all IF NOT EXISTS — safe to
             call on every pipeline run, including the very first.

Constraints:
    - Credentials are fetched from AWS Secrets Manager (JSON with host, port,
      dbname, username, password keys).
    - psycopg2-binary must be present in the container (embed_and_index.py
      pip-installs it before importing this module).
    - Raises ``PgVectorError`` on any database or Secrets Manager failure.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import boto3
import psycopg2
import psycopg2.extensions
from botocore.exceptions import BotoCoreError, ClientError

from chitrakatha.exceptions import PgVectorError
from chitrakatha.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# DDL executed on every run — all statements are idempotent.
_DDL_STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    """
    CREATE TABLE IF NOT EXISTS embeddings (
        id              BIGSERIAL PRIMARY KEY,
        source_document TEXT        NOT NULL,
        chunk_index     INT         NOT NULL,
        chunk_text      TEXT        NOT NULL,
        embedding       vector(1024),
        created_at      TIMESTAMPTZ DEFAULT NOW(),
        UNIQUE (source_document, chunk_index)
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
        ON embeddings
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
    """,
]


def _get_db_credentials(secret_arn: str, aws_region: str) -> dict[str, Any]:
    """Fetch RDS credentials JSON from Secrets Manager."""
    sm = boto3.client("secretsmanager", region_name=aws_region)
    try:
        resp = sm.get_secret_value(SecretId=secret_arn)
        return json.loads(resp["SecretString"])  # type: ignore[no-any-return]
    except (ClientError, BotoCoreError) as exc:
        raise PgVectorError(
            f"Failed to fetch DB credentials from {secret_arn}: {exc}"
        ) from exc


def _connect(creds: dict[str, Any]) -> psycopg2.extensions.connection:
    """Open a psycopg2 connection from a credentials dict."""
    try:
        return psycopg2.connect(
            host=creds["host"],
            port=int(creds["port"]),
            dbname=creds["dbname"],
            user=creds["username"],
            password=creds["password"],
            connect_timeout=10,
            sslmode="require",
        )
    except psycopg2.Error as exc:
        raise PgVectorError(f"PostgreSQL connection failed: {exc}") from exc


def init_schema(conn: psycopg2.extensions.connection) -> None:
    """Create pgvector extension, embeddings table, and HNSW index if not present.

    Safe to call on every pipeline run — all DDL uses IF NOT EXISTS.
    """
    try:
        with conn.cursor() as cur:
            for statement in _DDL_STATEMENTS:
                cur.execute(statement)
        conn.commit()
    except psycopg2.Error as exc:
        conn.rollback()
        raise PgVectorError(f"Schema init failed: {exc}") from exc


def write_vectors(
    chunk_embeddings: list[tuple[Chunk, list[float]]],
    db_secret_arn: str,
    aws_region: str,
    conn: psycopg2.extensions.connection | None = None,
) -> int:
    """Write chunk embeddings to the pgvector embeddings table.

    Args:
        chunk_embeddings: List of (Chunk, embedding) tuples from embed_chunks().
        db_secret_arn: Secrets Manager ARN holding RDS credentials JSON.
        aws_region: AWS region for the Secrets Manager client.
        conn: Optional pre-built psycopg2 connection (injected for testing).

    Returns:
        Number of rows newly inserted (0 if all already existed).

    Raises:
        PgVectorError: On Secrets Manager or PostgreSQL failure.
    """
    if not chunk_embeddings:
        return 0

    _conn = conn
    _owns_conn = _conn is None
    if _conn is None:
        creds = _get_db_credentials(db_secret_arn, aws_region)
        _conn = _connect(creds)

    try:
        init_schema(_conn)

        inserted = 0
        with _conn.cursor() as cur:
            for chunk, embedding in chunk_embeddings:
                cur.execute(
                    """
                    INSERT INTO embeddings
                        (source_document, chunk_index, chunk_text, embedding)
                    VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (source_document, chunk_index) DO NOTHING
                    """,
                    (
                        chunk.source_document,
                        chunk.chunk_index,
                        chunk.text,
                        str(embedding),  # pgvector accepts '[f1,f2,...]' string form
                    ),
                )
                if cur.rowcount > 0:
                    inserted += 1
        _conn.commit()

        logger.info(
            "pgvector: inserted %d new row(s), skipped %d duplicate(s).",
            inserted,
            len(chunk_embeddings) - inserted,
        )
        return inserted

    except psycopg2.Error as exc:
        _conn.rollback()
        raise PgVectorError(f"pgvector write failed: {exc}") from exc

    finally:
        if _owns_conn:
            _conn.close()
