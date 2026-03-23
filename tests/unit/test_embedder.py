"""Unit tests for the Bedrock Titan Embed v2 embedder.

All Bedrock API calls are mocked via ``unittest.mock.patch`` — no real AWS
calls are made. Tests verify batching logic, dimension validation, retry
behaviour, and error propagation.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, call, patch

import pytest
from botocore.exceptions import ClientError

from chitrakatha.exceptions import BedrockEmbeddingError
from chitrakatha.ingestion.chunker import Chunk
from chitrakatha.ingestion.embedder import (
    _EXPECTED_DIM,
    _MAX_BATCH_SIZE,
    embed_chunks,
    embed_query,
)


def _make_chunk(text: str, index: int = 0) -> Chunk:
    """Helper: build a minimal Chunk for testing."""
    return Chunk(
        text=text,
        token_count=len(text.split()),
        chunk_index=index,
        source_document="test.txt",
    )


def _mock_bedrock_response(embedding: list[float]) -> MagicMock:
    """Return a mock boto3 response for a successful Bedrock InvokeModel call."""
    body_mock = MagicMock()
    body_mock.read.return_value = json.dumps({"embedding": embedding}).encode()
    response_mock = MagicMock()
    response_mock.__getitem__ = MagicMock(side_effect={"body": body_mock}.__getitem__)
    return response_mock


class TestEmbedChunks:
    """Tests for ``embed_chunks``."""

    def test_single_chunk_produces_one_embedding(self) -> None:
        """A single chunk produces exactly one (chunk, embedding) result."""
        chunk = _make_chunk("Nagraj is a superhero.")
        fake_emb = [0.1] * _EXPECTED_DIM

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _mock_bedrock_response(fake_emb)

        results = embed_chunks([chunk], aws_region="ap-southeast-2", bedrock_client=mock_client)
        print(f"\n  [TEST] Single Chunk Embedding: Received {len(results[0][1])}-dim vector for chunk '{chunk.text[:10]}...'")

        assert len(results) == 1
        returned_chunk, embedding = results[0]
        assert returned_chunk is chunk
        assert len(embedding) == _EXPECTED_DIM
        assert embedding == fake_emb

    def test_batching_respects_max_batch_size(self) -> None:
        """More than MAX_BATCH_SIZE chunks triggers multiple invoke_model calls."""
        n_chunks = _MAX_BATCH_SIZE + 5  # e.g., 30 chunks
        chunks = [_make_chunk(f"text chunk {i}", i) for i in range(n_chunks)]
        fake_emb = [0.0] * _EXPECTED_DIM

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _mock_bedrock_response(fake_emb)

        results = embed_chunks(chunks, aws_region="ap-southeast-2", bedrock_client=mock_client)

        assert len(results) == n_chunks
        assert mock_client.invoke_model.call_count == n_chunks

    def test_wrong_dimension_raises(self) -> None:
        """A vector with unexpected dimension raises ``BedrockEmbeddingError``."""
        chunk = _make_chunk("Some text.")
        wrong_dim_emb = [0.1] * 512  # Wrong: should be 1024

        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _mock_bedrock_response(wrong_dim_emb)

        with pytest.raises(BedrockEmbeddingError, match="1024"):
            embed_chunks([chunk], aws_region="ap-southeast-2", bedrock_client=mock_client)

    def test_empty_chunks_raises_value_error(self) -> None:
        """Passing an empty list raises ``ValueError``."""
        with pytest.raises(ValueError, match="empty chunk list"):
            embed_chunks([], aws_region="ap-southeast-2")

    def test_throttling_retries_and_succeeds(self) -> None:
        """ThrottlingException on first call is retried; success on second call."""
        chunk = _make_chunk("Retry test text.")
        fake_emb = [0.5] * _EXPECTED_DIM

        throttle_error = ClientError(
            {"Error": {"Code": "ThrottlingException", "Message": "Rate exceeded"}},
            "InvokeModel",
        )
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = [
            throttle_error,
            _mock_bedrock_response(fake_emb),
        ]

        with patch("chitrakatha.ingestion.embedder.time.sleep"):  # Speed up test
            results = embed_chunks([chunk], aws_region="ap-southeast-2", bedrock_client=mock_client)

        assert len(results) == 1
        assert mock_client.invoke_model.call_count == 2

    def test_non_retryable_error_raises_immediately(self) -> None:
        """A non-throttling ClientError is not retried and raises immediately."""
        chunk = _make_chunk("Auth error test.")
        auth_error = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "InvokeModel",
        )
        mock_client = MagicMock()
        mock_client.invoke_model.side_effect = auth_error

        with pytest.raises(BedrockEmbeddingError, match="AccessDeniedException"):
            embed_chunks([chunk], aws_region="ap-southeast-2", bedrock_client=mock_client)

        assert mock_client.invoke_model.call_count == 1  # No retries


class TestEmbedQuery:
    """Tests for ``embed_query`` (live inference path)."""

    def test_query_returns_1024_dim_vector(self) -> None:
        """A valid query returns a 1024-dim embedding."""
        fake_emb = [0.2] * _EXPECTED_DIM
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _mock_bedrock_response(fake_emb)

        result = embed_query("Who is Nagraj?", aws_region="ap-southeast-2", bedrock_client=mock_client)
        print(f"  [TEST] Query Embedding: Generated {len(result)}-dim vector for user query.")

        assert len(result) == _EXPECTED_DIM

    def test_empty_query_raises(self) -> None:
        """Empty query string raises ``BedrockEmbeddingError`` without API call."""
        with pytest.raises(BedrockEmbeddingError, match="empty"):
            embed_query("", aws_region="ap-southeast-2")

    def test_devanagari_query_passes_through(self) -> None:
        """Hindi (Devanagari) query is passed to Bedrock without modification."""
        hindi_query = "नागराज कौन है?"
        fake_emb = [0.3] * _EXPECTED_DIM
        mock_client = MagicMock()
        mock_client.invoke_model.return_value = _mock_bedrock_response(fake_emb)

        result = embed_query(hindi_query, aws_region="ap-southeast-2", bedrock_client=mock_client)

        assert len(result) == _EXPECTED_DIM
        # Verify the query was passed without stripping Devanagari.
        call_body = json.loads(mock_client.invoke_model.call_args.kwargs["body"])
        assert hindi_query in call_body["inputText"]
