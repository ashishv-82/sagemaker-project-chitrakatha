"""Sliding-window text chunker for the Chitrakatha ingestion pipeline.

Why: A fixed overlap of 15% between consecutive chunks ensures that sentences
     spanning two chunk boundaries are represented in at least one chunk
     completely. This preserves narrative continuity between comic panels and
     dialogue sequences — critical for accurate RAG retrieval.

Design decisions:
    - Chunking is token-aware (splits on whitespace tokens, not bytes) so the
      chunk_size parameter maps predictably to LLM context windows.
    - Devanagari and other non-ASCII scripts are explicitly preserved — the
      chunker never strips, normalises-away, or truncates Unicode characters.
    - Returns typed ``Chunk`` pydantic v2 models so downstream steps
      (embedder, faiss_writer) benefit from runtime field validation.

Constraints:
    - ``overlap_ratio`` must be in [0.0, 0.5). Values >= 0.5 cause the chunk
      window to regress rather than advance.
    - Empty or whitespace-only text raises ``DataIngestionError`` rather than
      producing empty chunks.
"""

from __future__ import annotations

import unicodedata
import uuid
from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from chitrakatha.exceptions import DataIngestionError


class Chunk(BaseModel):
    """A single text chunk produced by the sliding-window chunker.

    Attributes:
        chunk_id: Unique identifier (UUID4) for deduplication in the vector store.
        text: The chunk text; UTF-8, Devanagari preserved.
        token_count: Approximate token count (whitespace-split word count).
        chunk_index: Zero-based position of this chunk in the source document.
        source_document: Filename or identifier of the originating document.
    """

    chunk_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    text: Annotated[str, Field(min_length=1)]
    token_count: int
    chunk_index: int
    source_document: str

    @field_validator("text")
    @classmethod
    def text_must_not_be_whitespace(cls, v: str) -> str:
        """Reject chunks that are entirely whitespace."""
        if not v.strip():
            raise ValueError("Chunk text must not be empty or whitespace-only.")
        return v


def _normalize_unicode(text: str) -> str:
    """Apply NFC Unicode normalisation while preserving Devanagari.

    NFC normalisation composes combining characters (e.g., ``अ`` + combining
    vowel) into their canonical precomposed form. This is safe for Devanagari
    and required for consistent token/character counting.

    Args:
        text: Raw input text (any Unicode).

    Returns:
        NFC-normalised text with all original characters intact.
    """
    return unicodedata.normalize("NFC", text)


def chunk_text(
    text: str,
    source_document: str,
    chunk_size: int = 512,
    overlap_ratio: float = 0.15,
) -> list[Chunk]:
    """Split ``text`` into overlapping chunks using a sliding window.

    The window advances by ``chunk_size * (1 - overlap_ratio)`` tokens at
    each step. The final partial chunk is included if it contains at least
    ``chunk_size // 4`` tokens (avoids tiny trailing fragments).

    Args:
        text: Source text to chunk. Must be non-empty UTF-8 (Devanagari safe).
        source_document: Identifier attached to each chunk for lineage tracking.
        chunk_size: Target number of whitespace-split tokens per chunk.
            Default 512 fits comfortably within Titan Embed v2's 8192-token limit.
        overlap_ratio: Fraction of ``chunk_size`` to overlap between consecutive
            chunks. Default 0.15 (15%) as specified by the project architecture.

    Returns:
        List of ``Chunk`` objects in source order. Empty list if text is blank.

    Raises:
        DataIngestionError: If ``text`` is empty/whitespace-only, or if
            ``overlap_ratio`` is outside [0.0, 0.5).
    """
    if not text or not text.strip():
        raise DataIngestionError(
            f"Cannot chunk document '{source_document}': text is empty or whitespace-only."
        )

    if not 0.0 <= overlap_ratio < 0.5:
        raise DataIngestionError(
            f"overlap_ratio must be in [0.0, 0.5), got {overlap_ratio}. "
            "Values >= 0.5 cause the window to regress."
        )

    text = _normalize_unicode(text)
    tokens: list[str] = text.split()

    if not tokens:
        raise DataIngestionError(
            f"Document '{source_document}' contains no tokens after normalisation."
        )

    step = max(1, int(chunk_size * (1 - overlap_ratio)))
    min_tail_tokens = max(1, chunk_size // 4)

    chunks: list[Chunk] = []
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        window_tokens = tokens[start:end]

        # Skip trailing micro-fragments (< 25% of chunk_size).
        if len(window_tokens) < min_tail_tokens and chunks:
            break

        chunk_text_str = " ".join(window_tokens)
        chunks.append(
            Chunk(
                text=chunk_text_str,
                token_count=len(window_tokens),
                chunk_index=chunk_index,
                source_document=source_document,
            )
        )
        chunk_index += 1
        start += step

    return chunks
