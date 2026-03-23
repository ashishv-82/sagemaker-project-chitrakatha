"""Unit tests for the Lambda API Gateway bridge (serving/lambda/handler.py).

All external calls (Bedrock, Secrets Manager, psycopg2) are mocked.
Tests cover:
    - English query → 200 + language="en".
    - Devanagari query → 200 + language="hi".
    - No chunks found → 200 with fallback answer.
    - Missing/invalid request body → 400.
    - RAG pipeline failure → 500.
    - Content-Type header always "application/json".
    - Direct Lambda invocation (no 'body' key in event).

Note: serving/lambda/ is a directory named 'lambda' (Python keyword).
sys.path manipulation is used to import handler directly.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import psycopg2
import pytest

# Add serving/lambda to sys.path so we can import 'handler' directly.
_LAMBDA_DIR = Path(__file__).parent.parent.parent / "serving" / "lambda"
if str(_LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(_LAMBDA_DIR))

import handler as lambda_handler  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_CREDS = {
    "host": "pg.test.internal",
    "port": 5432,
    "dbname": "chitrakatha",
    "username": "admin",
    "password": "secret",
}

_FAKE_EMBEDDING = [0.1] * 1024

_FAKE_CHUNKS = [
    ("Nagraj is a serpent superhero.", "raj_comics_1990.txt"),
    ("He first appeared in Nagraj comics.", "raj_comics_1990.txt"),
]


def _make_event(body: object) -> dict:
    """Wrap payload in an API Gateway proxy event envelope."""
    return {"body": json.dumps(body)}


def _make_sm_creds_response() -> MagicMock:
    mock = MagicMock()
    mock.get_secret_value.return_value = {
        "SecretString": json.dumps(_FAKE_CREDS)
    }
    return mock


def _make_bedrock_embed_response(embedding: list[float] = _FAKE_EMBEDDING) -> MagicMock:
    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps({"embedding": embedding}).encode()
    response_mock = MagicMock()
    response_mock.__getitem__ = MagicMock(side_effect={"body": body_mock}.__getitem__)
    return response_mock


def _make_bedrock_generate_response(text: str) -> MagicMock:
    return {
        "output": {
            "message": {
                "content": [{"text": text}]
            }
        }
    }


def _make_mock_conn(rows: list[tuple] | None = None) -> MagicMock:
    """Build a mock psycopg2 connection that returns given rows from fetchall."""
    mock_cursor = MagicMock()
    mock_cursor.fetchall.return_value = rows or []
    mock_conn = MagicMock()
    mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
    mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
    return mock_conn


def _patch_rag(
    chunks: list[tuple] | None = None,
    answer: str = "Nagraj is a serpent superhero.",
) -> tuple:
    """Return a context-manager tuple that patches all three external calls."""
    mock_sm = _make_sm_creds_response()
    mock_bedrock = MagicMock()
    mock_bedrock.invoke_model.return_value = _make_bedrock_embed_response()
    mock_bedrock.converse.return_value = _make_bedrock_generate_response(answer)
    mock_conn = _make_mock_conn(chunks if chunks is not None else _FAKE_CHUNKS)

    return mock_sm, mock_bedrock, mock_conn


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestValidRequests:
    def test_english_query_returns_200_with_en_language(self) -> None:
        mock_sm, mock_bedrock, mock_conn = _patch_rag()
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(_make_event({"query": "Who is Nagraj?"}), None)

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["language"] == "en"
        assert "answer" in body
        assert "sources" in body

    def test_devanagari_query_returns_200_with_hi_language(self) -> None:
        mock_sm, mock_bedrock, mock_conn = _patch_rag(answer="नागराज एक सुपरहीरो है।")
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(
                _make_event({"query": "नागराज कौन है?"}), None
            )

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["language"] == "hi"

    def test_mixed_query_with_devanagari_tagged_hi(self) -> None:
        mock_sm, mock_bedrock, mock_conn = _patch_rag()
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(
                _make_event({"query": "Tell me about Nagraj नागराज"}), None
            )

        assert json.loads(resp["body"])["language"] == "hi"

    def test_response_always_has_answer_sources_language(self) -> None:
        mock_sm, mock_bedrock, mock_conn = _patch_rag()
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(_make_event({"query": "Who is Doga?"}), None)

        body = json.loads(resp["body"])
        assert "answer" in body
        assert "sources" in body
        assert "language" in body

    def test_direct_lambda_invocation_no_body_key(self) -> None:
        """Direct invocation (no API GW envelope) — payload IS the event dict."""
        mock_sm, mock_bedrock, mock_conn = _patch_rag()
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler({"query": "Who is Phantom?"}, None)

        assert resp["statusCode"] == 200

    def test_no_chunks_returns_fallback_answer(self) -> None:
        """When pgvector returns no rows, handler returns a 200 with fallback text."""
        mock_sm, mock_bedrock, mock_conn = _patch_rag(chunks=[])
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(_make_event({"query": "Unknown topic"}), None)

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["sources"] == []
        assert "knowledge base" in body["answer"].lower()

    def test_sources_deduplicated(self) -> None:
        """Multiple chunks from same source_document produce one source entry."""
        duplicate_chunks = [
            ("Chunk A from same doc.", "raj_comics.txt"),
            ("Chunk B from same doc.", "raj_comics.txt"),
        ]
        mock_sm, mock_bedrock, mock_conn = _patch_rag(chunks=duplicate_chunks)
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(_make_event({"query": "test"}), None)

        body = json.loads(resp["body"])
        assert body["sources"] == ["raj_comics.txt"]


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestInvalidRequests:
    def test_missing_query_field_returns_400(self) -> None:
        resp = lambda_handler.handler(_make_event({"question": "Who is Nagraj?"}), None)
        assert resp["statusCode"] == 400
        assert "error" in json.loads(resp["body"])

    def test_invalid_json_body_returns_400(self) -> None:
        resp = lambda_handler.handler({"body": "not valid json {{{"}, None)
        assert resp["statusCode"] == 400
        assert "error" in json.loads(resp["body"])

    def test_empty_body_dict_returns_400(self) -> None:
        resp = lambda_handler.handler({"body": "{}"}, None)
        assert resp["statusCode"] == 400

    def test_rag_pipeline_failure_returns_500(self) -> None:
        mock_sm = _make_sm_creds_response()
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = Exception("Bedrock unreachable")

        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
        ):
            resp = lambda_handler.handler(_make_event({"query": "Who is Billu?"}), None)

        assert resp["statusCode"] == 500
        assert "error" in json.loads(resp["body"])

    def test_response_content_type_is_json(self) -> None:
        mock_sm, mock_bedrock, mock_conn = _patch_rag()
        with (
            patch.object(lambda_handler, "sm_client", mock_sm),
            patch.object(lambda_handler, "bedrock", mock_bedrock),
            patch("handler.psycopg2") as mock_psycopg2,
        ):
            mock_psycopg2.connect.return_value = mock_conn
            resp = lambda_handler.handler(_make_event({"query": "test"}), None)

        assert resp["headers"]["Content-Type"] == "application/json"

    def test_400_response_content_type_is_json(self) -> None:
        resp = lambda_handler.handler(_make_event({"not_query": "x"}), None)
        assert resp["headers"]["Content-Type"] == "application/json"
