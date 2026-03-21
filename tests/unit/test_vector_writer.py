"""Unit tests for the FAISS-on-S3 vector writer.

Why: The implementation pivoted from a native S3 Vectors API to a custom
     FAISS-on-S3 approach. Tests now verify S3 download/upload of the index
     and metadata, FAISS search logic, and idempotency using S3 mocks.
"""

from __future__ import annotations

import os
import pickle
import tempfile
from unittest.mock import MagicMock, patch, mock_open

import faiss
import numpy as np
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
    """Tests for ``write_vectors`` using FAISS + S3."""

    @patch("chitrakatha.ingestion.vector_writer._get_s3_client")
    @patch("faiss.write_index")
    @patch("faiss.read_index")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.dump")
    def test_writes_new_index_if_missing(self, mock_pickle, mock_file, mock_read, mock_write, mock_s3_factory) -> None:
        """If S3 404s, a new FAISS index is created and uploaded."""
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        # Fail the first download (index.faiss)
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "DownloadFile"
        )

        chunk = _make_chunk("Test text")
        emb = _make_embedding()

        print("\n  [TEST] S3 Cold Start: Verifying new index initialization when files are missing from S3...")
        written = write_vectors([(chunk, emb)], "bucket", "prefix", "ap-southeast-2")
        
        assert written == 1
        # Should have tried to download the index (and stopped after it 404'd)
        assert mock_s3.download_file.call_count == 1
        assert mock_write.called

    @patch("chitrakatha.ingestion.vector_writer._get_s3_client")
    @patch("faiss.read_index")
    @patch("faiss.write_index")
    @patch("builtins.open", new_callable=mock_open)
    @patch("pickle.load")
    @patch("pickle.dump")
    def test_idempotency_skips_existing_ids(
        self, mock_pickle_dump, mock_pickle_load, mock_file, mock_write, mock_read, mock_s3_factory
    ) -> None:
        """Chunks whose IDs are in ``existing_ids`` are not processed."""
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        
        # Mock index behavior
        mock_index = MagicMock()
        mock_index.ntotal = 0
        mock_read.return_value = mock_index
        mock_pickle_load.return_value = {}

        chunk1 = _make_chunk("First chunk.", 0)
        chunk2 = _make_chunk("Second chunk.", 1)
        emb = _make_embedding()

        print(f"  [TEST] Idempotency Control: Processing 2 blocks, skipping ID '{chunk1.chunk_id}'...")
        written = write_vectors(
            [(chunk1, emb), (chunk2, emb)],
            bucket_name="bucket",
            index_name="prefix",
            aws_region="ap-southeast-2",
            existing_ids={chunk1.chunk_id},
        )

        assert written == 1
        # Should only have added chunk2 (chunk1 was skipped via existing_ids)
        assert mock_index.add.called
        print(f"  [TEST] Success: Correctly filtered existing IDs. Written count: {written}")
        assert mock_s3.upload_file.call_count == 2 

    @patch("chitrakatha.ingestion.vector_writer._get_s3_client")
    def test_upload_failure_raises_s3_vector_error(self, mock_s3_factory) -> None:
        """ClientError during upload raises S3VectorError."""
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "DownloadFile"
        )
        mock_s3.upload_file.side_effect = ClientError(
            {"Error": {"Code": "AccessDenied", "Message": "Denied"}}, "UploadFile"
        )

        with pytest.raises(S3VectorError, match="AccessDenied"):
            write_vectors(
                [(_make_chunk(), _make_embedding())],
                bucket_name="bucket",
                index_name="prefix",
                aws_region="ap-southeast-2",
            )


class TestQueryVectors:
    """Tests for ``query_vectors`` using FAISS + S3."""

    @patch("chitrakatha.ingestion.vector_writer._get_s3_client")
    @patch("faiss.read_index")
    def test_query_returns_meta_results(self, mock_read, mock_s3_factory) -> None:
        """Mock index search returns indices that are mapped to metadata."""
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        
        # Mock index behavior
        mock_index = MagicMock()
        mock_index.search.return_value = (
            np.array([[0.9]]), # Scores
            np.array([[0]])    # Indices
        )
        mock_read.return_value = mock_index

        # Mock metadata file download
        def side_effect(bucket, key, path):
            if "metadata.pkl" in key:
                with open(path, "wb") as f:
                    pickle.dump({0: {"chunk_text": "Found me!"}}, f)
        mock_s3.download_file.side_effect = side_effect

        results = query_vectors(
            query_embedding=_make_embedding(),
            bucket_name="bucket",
            index_name="prefix",
            aws_region="ap-southeast-2",
        )

        assert len(results) == 1
        assert results[0]["chunk_text"] == "Found me!"
        assert results[0]["score"] == pytest.approx(0.9)

    @patch("chitrakatha.ingestion.vector_writer._get_s3_client")
    def test_query_missing_index_returns_empty(self, mock_s3_factory) -> None:
        """If index file is missing (404), return empty list."""
        mock_s3 = MagicMock()
        mock_s3_factory.return_value = mock_s3
        mock_s3.download_file.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}}, "DownloadFile"
        )

        results = query_vectors(
            query_embedding=_make_embedding(),
            bucket_name="bucket",
            index_name="prefix",
            aws_region="ap-southeast-2",
        )

        assert results == []
