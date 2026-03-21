"""Unit tests for the S3 Vectors writer.

All S3 Vectors API calls are mocked — no real AWS calls are made.
Tests verify batching, idempotency filtering, metadata construction,
and error propagation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, call

import pytest
from botocore.exceptions import ClientError

from chitrakatha.exceptions import S3VectorError
from chitrakatha.ingestion.chunker import Chunk
from chitrakatha.ingestion.vector_writer import write_vectors, query_vectors


def _make_chunk(text: str = "Nagraj is a superhero.", index: int = 0) -> Chunk:
    return Chunk(
        text=text,
        token_count=len(text.split()),
        chunk_index=index,
        source_document="test.txt",
    )


def _make_embedding() -> list[float]:
    return [0.1] * 1536


class TestWriteVectors:
    """Tests for ``write_vectors``."""

    def test_single_vector_written_successfully(self) -> None:
        """A single (chunk, embedding) pair results in one put_vectors call."""
        chunk = _make_chunk()
        emb = _make_embedding()
        mock_client = MagicMock()

        written = write_vectors(
            [(chunk, emb)],
            bucket_name="test-vectors-bucket",
            index_name="test-index",
            aws_region="ap-southeast-2",
            s3vectors_client=mock_client,
        )

        assert written == 1
        assert mock_client.put_vectors.call_count == 1
        call_kwargs = mock_client.put_vectors.call_args.kwargs
        assert call_kwargs["VectorBucketName"] == "test-vectors-bucket"
        assert call_kwargs["IndexName"] == "test-index"
        assert len(call_kwargs["Vectors"]) == 1

    def test_idempotency_skips_existing_ids(self) -> None:
        """Chunks whose IDs are in ``existing_ids`` are not written."""
        chunk1 = _make_chunk("First chunk.", 0)
        chunk2 = _make_chunk("Second chunk.", 1)
        emb = _make_embedding()
        mock_client = MagicMock()

        written = write_vectors(
            [(chunk1, emb), (chunk2, emb)],
            bucket_name="bucket",
            index_name="index",
            aws_region="ap-southeast-2",
            s3vectors_client=mock_client,
            existing_ids={chunk1.chunk_id},  # chunk1 already exists
        )

        assert written == 1  # Only chunk2 was written.
        written_keys = [
            v["Key"] for v in mock_client.put_vectors.call_args.kwargs["Vectors"]
        ]
        assert chunk2.chunk_id in written_keys
        assert chunk1.chunk_id not in written_keys

    def test_extra_metadata_attached_to_vectors(self) -> None:
        """Extra metadata fields (entity, publisher, language) appear in each vector."""
        chunk = _make_chunk()
        mock_client = MagicMock()

        write_vectors(
            [(chunk, _make_embedding())],
            bucket_name="bucket",
            index_name="index",
            aws_region="ap-southeast-2",
            s3vectors_client=mock_client,
            extra_metadata={"source_entity": "Nagraj", "publisher": "Raj Comics", "language": "en"},
        )

        vector_meta = mock_client.put_vectors.call_args.kwargs["Vectors"][0]["Metadata"]
        assert vector_meta["source_entity"] == "Nagraj"
        assert vector_meta["publisher"] == "Raj Comics"
        assert vector_meta["language"] == "en"
        assert vector_meta["chunk_text"] == chunk.text  # chunk_text always included

    def test_batch_write_splits_correctly(self) -> None:
        """101 vectors are written in 2 put_vectors calls (max batch = 100)."""
        pairs = [(_make_chunk(f"text {i}", i), _make_embedding()) for i in range(101)]
        mock_client = MagicMock()

        written = write_vectors(
            pairs,
            bucket_name="bucket",
            index_name="index",
            aws_region="ap-southeast-2",
            s3vectors_client=mock_client,
        )

        assert written == 101
        assert mock_client.put_vectors.call_count == 2

    def test_api_failure_raises_s3_vector_error(self) -> None:
        """A ClientError from put_vectors raises ``S3VectorError``."""
        chunk = _make_chunk()
        mock_client = MagicMock()
        mock_client.put_vectors.side_effect = ClientError(
            {"Error": {"Code": "AccessDeniedException", "Message": "Denied"}},
            "PutVectors",
        )

        with pytest.raises(S3VectorError, match="AccessDeniedException"):
            write_vectors(
                [(chunk, _make_embedding())],
                bucket_name="bucket",
                index_name="index",
                aws_region="ap-southeast-2",
                s3vectors_client=mock_client,
            )

    def test_empty_input_raises_value_error(self) -> None:
        """Passing an empty list raises ``ValueError``."""
        with pytest.raises(ValueError, match="empty"):
            write_vectors([], bucket_name="b", index_name="i", aws_region="ap-southeast-2")


class TestQueryVectors:
    """Tests for ``query_vectors``."""

    def test_returns_top_k_results(self) -> None:
        """Response is parsed into a list of result dicts with score and metadata."""
        mock_client = MagicMock()
        mock_client.query_vectors.return_value = {
            "Vectors": [
                {"Score": 0.95, "Metadata": {"chunk_text": "Nagraj fact.", "source_entity": "Nagraj"}},
                {"Score": 0.80, "Metadata": {"chunk_text": "Dhruva fact.", "source_entity": "Dhruva"}},
            ]
        }

        results = query_vectors(
            query_embedding=[0.1] * 1536,
            bucket_name="bucket",
            index_name="index",
            aws_region="ap-southeast-2",
            top_k=2,
            s3vectors_client=mock_client,
        )

        assert len(results) == 2
        assert results[0]["score"] == 0.95
        assert results[0]["chunk_text"] == "Nagraj fact."
        assert results[1]["source_entity"] == "Dhruva"

    def test_api_failure_raises_s3_vector_error(self) -> None:
        """A ClientError from query_vectors raises ``S3VectorError``."""
        mock_client = MagicMock()
        mock_client.query_vectors.side_effect = ClientError(
            {"Error": {"Code": "ServiceUnavailableException", "Message": "Down"}},
            "QueryVectors",
        )

        with pytest.raises(S3VectorError, match="ServiceUnavailableException"):
            query_vectors(
                query_embedding=[0.1] * 1536,
                bucket_name="bucket",
                index_name="index",
                aws_region="ap-southeast-2",
                s3vectors_client=mock_client,
            )
