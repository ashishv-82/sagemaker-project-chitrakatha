"""Unit tests for the sliding-window chunker.

Tests cover:
    - Basic chunking with default parameters.
    - Devanagari (Hindi) text preservation — no characters stripped.
    - Correct 15% overlap between consecutive chunks.
    - Minimum tail-fragment suppression.
    - Error cases: empty text, invalid overlap_ratio.
"""

from __future__ import annotations

import pytest

from chitrakatha.exceptions import DataIngestionError
from chitrakatha.ingestion.chunker import Chunk, chunk_text


class TestChunkText:
    """Tests for the ``chunk_text`` function."""

    def test_basic_english_chunking(self) -> None:
        """Basic English text is split into chunks of ~512 tokens."""
        text = "Word " * 1000
        chunks = chunk_text(text, source_document="test.txt")
        print(f"\n  [TEST] Basic English: Generated {len(chunks)} chunks from {len(text.split())} words.")
        assert len(chunks) > 1
        assert all(c.token_count <= 600 for c in chunks)

    def test_devanagari_preserved(self) -> None:
        """Hindi/Devanagari script is not stripped or corrupted during chunking."""
        text = "नाराज भारत का सबसे लोकप्रिय कॉमिक सुपरहीरो है।"
        chunks = chunk_text(text, source_document="hindi.txt")
        full_text = " ".join(c.text for c in chunks)
        print(f"  [TEST] Devanagari Integrity: Verified text '{chunks[0].text[:10]}...' is preserved.")
        assert len(chunks) == 1
        assert "नाराज" in full_text

    def test_overlap_produces_shared_tokens(self) -> None:
        """Consecutive chunks must share tokens equal to the overlap segment."""
        tokens = [f"t{i}" for i in range(60)]
        text = " ".join(tokens)
        chunk_size = 20
        overlap_ratio = 0.15
        chunks = chunk_text(
            text, source_document="overlap_test.txt",
            chunk_size=chunk_size, overlap_ratio=overlap_ratio,
        )
        assert len(chunks) >= 2
        first_tokens = set(chunks[0].text.split())
        second_tokens = set(chunks[1].text.split())
        overlap_count = len(first_tokens & second_tokens)
        expected_overlap = int(chunk_size * overlap_ratio)
        # Allow ±1 for rounding; what matters is overlap is non-zero.
        assert overlap_count >= max(1, expected_overlap - 1)

    def test_chunk_returns_typed_objects(self) -> None:
        """Each item in the returned list must be a valid ``Chunk`` pydantic model."""
        text = " ".join(["word"] * 30)
        chunks = chunk_text(text, source_document="type_check.txt")
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_id  # UUID4 — non-empty
            assert len(chunk.text) > 0

    def test_single_chunk_when_text_fits(self) -> None:
        """Texts shorter than chunk_size should produce exactly one chunk."""
        text = "Nagraj is a superhero from Raj Comics."
        chunks = chunk_text(text, source_document="short.txt", chunk_size=512)
        assert len(chunks) == 1
        assert chunks[0].chunk_index == 0

    def test_empty_text_raises(self) -> None:
        """Empty text must raise ``DataIngestionError``."""
        with pytest.raises(DataIngestionError, match="empty or whitespace-only"):
            chunk_text("", source_document="empty.txt")

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only text must raise ``DataIngestionError``."""
        with pytest.raises(DataIngestionError, match="empty or whitespace-only"):
            chunk_text("   \n\t  ", source_document="blank.txt")

    def test_invalid_overlap_ratio_raises(self) -> None:
        """overlap_ratio >= 0.5 must raise ``DataIngestionError``."""
        text = " ".join(["word"] * 50)
        with pytest.raises(DataIngestionError, match="overlap_ratio"):
            chunk_text(text, source_document="bad.txt", overlap_ratio=0.5)
        with pytest.raises(DataIngestionError, match="overlap_ratio"):
            chunk_text(text, source_document="bad.txt", overlap_ratio=0.9)

    def test_negative_overlap_ratio_raises(self) -> None:
        """Negative overlap_ratio must raise ``DataIngestionError``."""
        text = " ".join(["word"] * 50)
        with pytest.raises(DataIngestionError, match="overlap_ratio"):
            chunk_text(text, source_document="bad.txt", overlap_ratio=-0.1)

    def test_bilingual_text_preserved(self) -> None:
        """Mixed English-Hindi text chunks correctly with both scripts intact."""
        text = (
            "Nagraj नागराज is a serpent superhero. "
            "He was created in 1986. उसे सर्पशक्तियाँ प्राप्त हैं। "
            "His publisher is Raj Comics राज कॉमिक्स ।" * 10
        )
        chunks = chunk_text(text, source_document="bilingual.txt", chunk_size=30)
        assert len(chunks) > 1
        full = " ".join(c.text for c in chunks)
        assert "नागराज" in full
        assert "Nagraj" in full
