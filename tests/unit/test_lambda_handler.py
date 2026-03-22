"""Unit tests for the Lambda API Gateway bridge (serving/lambda/handler.py).

Tests cover:
    - Valid English query: 200 + language="en" in response.
    - Valid Devanagari query: 200 + language="hi" in response.
    - Missing/invalid JSON body: 400 with error details.
    - Pydantic validation error (missing 'query' field): 400.
    - SageMaker endpoint invocation failure: 500.
    - Direct Lambda invocation (no 'body' key): treated as direct payload.
    - Response fields: answer, sources, language always present on success.

Note on import: serving/lambda/handler.py lives in a directory named 'lambda',
which is a Python keyword. sys.path manipulation is used to import it directly
as 'handler', avoiding the keyword conflict.
"""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# Add serving/lambda to sys.path so we can import 'handler' directly.
_LAMBDA_DIR = Path(__file__).parent.parent.parent / "serving" / "lambda"
if str(_LAMBDA_DIR) not in sys.path:
    sys.path.insert(0, str(_LAMBDA_DIR))

import handler as lambda_handler  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_apigw_event(body: Any) -> dict:
    """Wrap a payload in an API Gateway proxy event envelope."""
    return {"body": json.dumps(body)}


def _make_sm_response(answer: str, sources: list[str] | None = None) -> MagicMock:
    """Build a mock SageMaker invoke_endpoint response."""
    payload = {"answer": answer, "sources": sources or ["doc1.txt"]}
    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps(payload).encode("utf-8")
    response_mock = MagicMock()
    response_mock.__getitem__ = MagicMock(
        side_effect={"Body": body_mock}.__getitem__
    )
    return response_mock


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

class TestValidRequests:
    def test_english_query_returns_200_with_en_language(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response("Nagraj is a serpent superhero.")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            resp = lambda_handler.handler(_make_apigw_event({"query": "Who is Nagraj?"}), None)

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["language"] == "en"
        assert body["answer"] == "Nagraj is a serpent superhero."
        assert "sources" in body

    def test_devanagari_query_returns_200_with_hi_language(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response("नागराज एक सुपरहीरो है।")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            resp = lambda_handler.handler(
                _make_apigw_event({"query": "नागराज कौन है?"}), None
            )

        assert resp["statusCode"] == 200
        body = json.loads(resp["body"])
        assert body["language"] == "hi"

    def test_mixed_query_with_devanagari_tagged_hi(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response("Nagraj नागराज is a hero.")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            resp = lambda_handler.handler(
                _make_apigw_event({"query": "Tell me about Nagraj नागराज"}), None
            )

        assert resp["statusCode"] == 200
        assert json.loads(resp["body"])["language"] == "hi"

    def test_response_always_has_answer_sources_language(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response(
            "Doga is an anti-hero.", sources=["doga.txt", "raj_comics.txt"]
        )

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            resp = lambda_handler.handler(_make_apigw_event({"query": "Who is Doga?"}), None)

        body = json.loads(resp["body"])
        assert "answer" in body
        assert "sources" in body
        assert "language" in body
        assert len(body["sources"]) == 2

    def test_direct_lambda_invocation_no_body_key(self) -> None:
        """Direct invocation (no API GW envelope) — payload IS the event."""
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response("Phantom is Indrajal's hero.")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            # No 'body' key — handler falls back to json.dumps(event)
            resp = lambda_handler.handler({"query": "Who is Phantom?"}, None)

        assert resp["statusCode"] == 200

    def test_sagemaker_endpoint_name_forwarded(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response("Answer here.")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            lambda_handler.handler(_make_apigw_event({"query": "test"}), None)

        call_kwargs = mock_sm.invoke_endpoint.call_args.kwargs
        assert call_kwargs["EndpointName"] == lambda_handler.ENDPOINT_NAME
        assert call_kwargs["ContentType"] == "application/json"


# ---------------------------------------------------------------------------
# Error-path tests
# ---------------------------------------------------------------------------

class TestInvalidRequests:
    def test_missing_query_field_returns_400(self) -> None:
        resp = lambda_handler.handler(_make_apigw_event({"question": "Who is Nagraj?"}), None)

        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "error" in body

    def test_invalid_json_body_returns_400(self) -> None:
        event = {"body": "not valid json {{{"}
        resp = lambda_handler.handler(event, None)

        assert resp["statusCode"] == 400
        body = json.loads(resp["body"])
        assert "error" in body

    def test_empty_body_string_returns_400(self) -> None:
        resp = lambda_handler.handler({"body": "{}"}, None)
        # Empty dict has no 'query' field → Pydantic validation error → 400
        assert resp["statusCode"] == 400

    def test_sagemaker_failure_returns_500(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.side_effect = Exception("SageMaker unreachable")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            resp = lambda_handler.handler(_make_apigw_event({"query": "Who is Billu?"}), None)

        assert resp["statusCode"] == 500
        body = json.loads(resp["body"])
        assert "error" in body

    def test_response_content_type_is_json(self) -> None:
        mock_sm = MagicMock()
        mock_sm.invoke_endpoint.return_value = _make_sm_response("Test answer.")

        with patch.object(lambda_handler, "sm_runtime", mock_sm):
            resp = lambda_handler.handler(_make_apigw_event({"query": "test"}), None)

        assert resp["headers"]["Content-Type"] == "application/json"

    def test_400_response_content_type_is_json(self) -> None:
        resp = lambda_handler.handler(_make_apigw_event({"not_query": "x"}), None)
        assert resp["headers"]["Content-Type"] == "application/json"
