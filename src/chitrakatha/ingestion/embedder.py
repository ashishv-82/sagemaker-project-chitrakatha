"""Bedrock Titan Embed Text v2 wrapper for the Chitrakatha ingestion pipeline.

Why: Titan Embed Text v2 is used for both corpus embedding (Flow A, offline)
     and live query embedding at inference time (serving/inference.py). Wrapping
     it here provides a single, tested interface to both callers.

     Batching (max 25 per call) is critical: the Bedrock API has per-call
     throughput quotas, and batching reduces latency and cost versus one-call-
     per-chunk approaches.

Constraints:
    - Max batch size: 25 (Bedrock Titan Embed v2 API limit).
    - Output dimension: 1024 (Titan Embed v2 default; v2 does not support 1536).
    - Raises ``BedrockEmbeddingError`` on any API failure — never swallows errors
      silently, as a partial embedding batch would corrupt the vector index.
    - Retry logic: 3 attempts with exponential backoff on transient throttling.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Final

import boto3
from botocore.exceptions import BotoCoreError, ClientError

from chitrakatha.exceptions import BedrockEmbeddingError
from chitrakatha.ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Titan Embed Text v2 model identifier.
_TITAN_MODEL_ID: Final[str] = "amazon.titan-embed-text-v2:0"

# Maximum chunks per Bedrock API call (API hard limit).
_MAX_BATCH_SIZE: Final[int] = 25

# Expected output vector dimensionality.
# Titan Embed Text v2 supports 256, 512, or 1024 dims (default: 1024).
# Titan Embed Text v1 outputs 1536 dims — v2 does NOT support 1536.
_EXPECTED_DIM: Final[int] = 1024

# Retry configuration for transient throttling / service errors.
_MAX_RETRIES: Final[int] = 3
_RETRY_BASE_DELAY_SEC: Final[float] = 1.0


def _embed_single(
    text: str,
    bedrock_client: boto3.client,
    retries: int = _MAX_RETRIES,
) -> list[float]:
    """Embed a single text string via Bedrock Titan Embed v2.

    Args:
        text: Input text (UTF-8, Devanagari safe).
        bedrock_client: Pre-built ``bedrock-runtime`` boto3 client.
        retries: Number of retry attempts on throttling errors.

    Returns:
        List of 1024 floats representing the embedding vector.

    Raises:
        BedrockEmbeddingError: After all retries are exhausted or on a
            non-retryable API error.
    """
    payload = json.dumps({"inputText": text})

    for attempt in range(1, retries + 1):
        try:
            response = bedrock_client.invoke_model(
                modelId=_TITAN_MODEL_ID,
                contentType="application/json",
                accept="application/json",
                body=payload,
            )
            body = json.loads(response["body"].read())
            embedding: list[float] = body["embedding"]

            if len(embedding) != _EXPECTED_DIM:
                raise BedrockEmbeddingError(
                    f"Titan Embed v2 returned {len(embedding)}-dim vector; "
                    f"expected {_EXPECTED_DIM}. Model may have changed."
                )
            return embedding

        except ClientError as exc:
            error_code = exc.response["Error"]["Code"]
            if error_code in {"ThrottlingException", "ServiceUnavailableException"}:
                if attempt < retries:
                    delay = _RETRY_BASE_DELAY_SEC * (2 ** (attempt - 1))
                    logger.warning(
                        "Bedrock throttled (attempt %d/%d). Retrying in %.1fs.",
                        attempt, retries, delay,
                    )
                    time.sleep(delay)
                    continue
            raise BedrockEmbeddingError(
                f"Bedrock InvokeModel failed [{error_code}]: {exc}"
            ) from exc

        except BotoCoreError as exc:
            raise BedrockEmbeddingError(
                f"BotoCoreError during Bedrock embedding: {exc}"
            ) from exc

    raise BedrockEmbeddingError(
        f"Bedrock embedding failed after {retries} retries for text: {text[:80]}..."
    )


def embed_chunks(
    chunks: list[Chunk],
    aws_region: str,
    bedrock_client: boto3.client | None = None,
) -> list[tuple[Chunk, list[float]]]:
    """Embed a list of ``Chunk`` objects in batches of up to 25.

    Args:
        chunks: Chunks to embed. Must be non-empty.
        aws_region: AWS region for the Bedrock runtime endpoint.
        bedrock_client: Optional pre-built ``bedrock-runtime`` client
            (injected for testing via moto).

    Returns:
        List of ``(chunk, embedding)`` tuples in the same order as ``chunks``.

    Raises:
        BedrockEmbeddingError: If any single embedding call fails after retries.
        ValueError: If ``chunks`` is empty.
    """
    if not chunks:
        raise ValueError("embed_chunks received an empty chunk list.")

    client = bedrock_client or boto3.client(
        "bedrock-runtime", region_name=aws_region
    )

    results: list[tuple[Chunk, list[float]]] = []

    # Process in batches to respect API rate limits.
    for batch_start in range(0, len(chunks), _MAX_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + _MAX_BATCH_SIZE]
        logger.info(
            "Embedding batch %d–%d of %d chunks.",
            batch_start + 1,
            batch_start + len(batch),
            len(chunks),
        )
        for chunk in batch:
            embedding = _embed_single(chunk.text, client)
            results.append((chunk, embedding))

    return results


def embed_query(
    query_text: str,
    aws_region: str,
    bedrock_client: boto3.client | None = None,
) -> list[float]:
    """Embed a single user query string for RAG retrieval.

    This is the live-path equivalent of ``embed_chunks`` — called by
    ``serving/inference.py`` on every user request.

    Args:
        query_text: The user's raw query (English or Devanagari).
        aws_region: AWS region for the Bedrock runtime endpoint.
        bedrock_client: Optional pre-built client (for testing).

    Returns:
        1536-dim embedding vector.

    Raises:
        BedrockEmbeddingError: On API failure.
    """
    if not query_text or not query_text.strip():
        raise BedrockEmbeddingError("Query text must not be empty.")

    client = bedrock_client or boto3.client(
        "bedrock-runtime", region_name=aws_region
    )
    return _embed_single(query_text, client)
