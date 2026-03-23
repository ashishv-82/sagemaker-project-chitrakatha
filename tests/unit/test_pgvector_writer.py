"""Unit tests for the pgvector writer.

All PostgreSQL and AWS Secrets Manager calls are mocked — no real connections.
Tests cover schema init, successful inserts, conflict skipping, credential
fetching, and error propagation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import psycopg2
import pytest
from botocore.exceptions import ClientError

from chitrakatha.exceptions import PgVectorError
from chitrakatha.ingestion.chunker import Chunk
from chitrakatha.ingestion.pgvector_writer import (
    _DDL_STATEMENTS,
    _connect,
    _get_db_credentials,
    init_schema,
    write_vectors,
)

_FAKE_CREDS = {
    "host": "pg.example.com",
    "port": 5432,
    "dbname": "chitrakatha",
    "username": "admin",
    "password": "s3cret",
}

_FAKE_SECRET_ARN = "arn:aws:secretsmanager:ap-southeast-2:123456789012:secret:chitrakatha/rds_credentials"
_FAKE_REGION = "ap-southeast-2"


def _make_chunk(text: str = "hello world", index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        token_count=len(text.split()),
        chunk_index=index,
        source_document="test_doc.txt",
    )


def _make_embedding(dim: int = 1024) -> list[float]:
    return [0.1] * dim


# ---------------------------------------------------------------------------
# _get_db_credentials
# ---------------------------------------------------------------------------

def test_get_db_credentials_success() -> None:
    mock_sm = MagicMock()
    mock_sm.get_secret_value.return_value = {
        "SecretString": json.dumps(_FAKE_CREDS)
    }
    with patch("chitrakatha.ingestion.pgvector_writer.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sm
        result = _get_db_credentials(_FAKE_SECRET_ARN, _FAKE_REGION)

    assert result == _FAKE_CREDS
    mock_boto3.client.assert_called_once_with("secretsmanager", region_name=_FAKE_REGION)


def test_get_db_credentials_client_error_raises_pgvector_error() -> None:
    mock_sm = MagicMock()
    mock_sm.get_secret_value.side_effect = ClientError(
        {"Error": {"Code": "ResourceNotFoundException", "Message": "not found"}},
        "GetSecretValue",
    )
    with patch("chitrakatha.ingestion.pgvector_writer.boto3") as mock_boto3:
        mock_boto3.client.return_value = mock_sm
        with pytest.raises(PgVectorError, match="Failed to fetch DB credentials"):
            _get_db_credentials(_FAKE_SECRET_ARN, _FAKE_REGION)


# ---------------------------------------------------------------------------
# _connect
# ---------------------------------------------------------------------------

def test_connect_success() -> None:
    mock_conn = MagicMock()
    with patch("chitrakatha.ingestion.pgvector_writer.psycopg2") as mock_psycopg2:
        mock_psycopg2.connect.return_value = mock_conn
        mock_psycopg2.Error = psycopg2.Error
        result = _connect(_FAKE_CREDS)

    assert result is mock_conn
    mock_psycopg2.connect.assert_called_once_with(
        host="pg.example.com",
        port=5432,
        dbname="chitrakatha",
        user="admin",
        password="s3cret",
        connect_timeout=10,
        sslmode="require",
    )


def test_connect_failure_raises_pgvector_error() -> None:
    with patch("chitrakatha.ingestion.pgvector_writer.psycopg2") as mock_psycopg2:
        mock_psycopg2.Error = psycopg2.Error
        mock_psycopg2.connect.side_effect = psycopg2.OperationalError("refused")
        with pytest.raises(PgVectorError, match="PostgreSQL connection failed"):
            _connect(_FAKE_CREDS)


# ---------------------------------------------------------------------------
# init_schema
# ---------------------------------------------------------------------------

def test_init_schema_executes_all_ddl() -> None:
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    init_schema(mock_conn)

    assert mock_cursor.execute.call_count == len(_DDL_STATEMENTS)
    mock_conn.commit.assert_called_once()


def test_init_schema_rolls_back_on_error() -> None:
    mock_conn = MagicMock()
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = psycopg2.ProgrammingError("syntax error")
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    with pytest.raises(PgVectorError, match="Schema init failed"):
        init_schema(mock_conn)

    mock_conn.rollback.assert_called_once()


# ---------------------------------------------------------------------------
# write_vectors
# ---------------------------------------------------------------------------

def _make_mock_conn(rowcount: int = 1) -> MagicMock:
    """Build a mock psycopg2 connection where each execute produces rowcount rows."""
    mock_cursor = MagicMock()
    mock_cursor.rowcount = rowcount
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


def test_write_vectors_returns_zero_for_empty_input() -> None:
    result = write_vectors([], _FAKE_SECRET_ARN, _FAKE_REGION)
    assert result == 0


def test_write_vectors_inserts_new_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_conn = _make_mock_conn(rowcount=1)

    # Stub out init_schema and _get_db_credentials / _connect to isolate the insert
    monkeypatch.setattr("chitrakatha.ingestion.pgvector_writer.init_schema", MagicMock())
    monkeypatch.setattr(
        "chitrakatha.ingestion.pgvector_writer._get_db_credentials",
        MagicMock(return_value=_FAKE_CREDS),
    )
    monkeypatch.setattr(
        "chitrakatha.ingestion.pgvector_writer._connect",
        MagicMock(return_value=mock_conn),
    )

    chunk_embeddings = [(_make_chunk("text one", 0), _make_embedding())]
    result = write_vectors(chunk_embeddings, _FAKE_SECRET_ARN, _FAKE_REGION)

    assert result == 1
    mock_conn.commit.assert_called_once()
    mock_conn.close.assert_called_once()


def test_write_vectors_skips_conflicts(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_conn = _make_mock_conn(rowcount=0)  # ON CONFLICT DO NOTHING → rowcount=0

    monkeypatch.setattr("chitrakatha.ingestion.pgvector_writer.init_schema", MagicMock())
    monkeypatch.setattr(
        "chitrakatha.ingestion.pgvector_writer._get_db_credentials",
        MagicMock(return_value=_FAKE_CREDS),
    )
    monkeypatch.setattr(
        "chitrakatha.ingestion.pgvector_writer._connect",
        MagicMock(return_value=mock_conn),
    )

    chunk_embeddings = [(_make_chunk("existing chunk", 0), _make_embedding())]
    result = write_vectors(chunk_embeddings, _FAKE_SECRET_ARN, _FAKE_REGION)

    assert result == 0  # nothing newly inserted
    mock_conn.commit.assert_called_once()


def test_write_vectors_uses_provided_conn(monkeypatch: pytest.MonkeyPatch) -> None:
    """When conn is passed in, write_vectors must NOT close it."""
    mock_conn = _make_mock_conn(rowcount=1)

    monkeypatch.setattr("chitrakatha.ingestion.pgvector_writer.init_schema", MagicMock())

    chunk_embeddings = [(_make_chunk(), _make_embedding())]
    write_vectors(chunk_embeddings, _FAKE_SECRET_ARN, _FAKE_REGION, conn=mock_conn)

    mock_conn.close.assert_not_called()


def test_write_vectors_rolls_back_on_db_error(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_cursor = MagicMock()
    mock_cursor.execute.side_effect = psycopg2.OperationalError("connection lost")
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)

    monkeypatch.setattr("chitrakatha.ingestion.pgvector_writer.init_schema", MagicMock())
    monkeypatch.setattr(
        "chitrakatha.ingestion.pgvector_writer._get_db_credentials",
        MagicMock(return_value=_FAKE_CREDS),
    )
    monkeypatch.setattr(
        "chitrakatha.ingestion.pgvector_writer._connect",
        MagicMock(return_value=mock_conn),
    )

    with pytest.raises(PgVectorError, match="pgvector write failed"):
        write_vectors([(_make_chunk(), _make_embedding())], _FAKE_SECRET_ARN, _FAKE_REGION)

    mock_conn.rollback.assert_called_once()
