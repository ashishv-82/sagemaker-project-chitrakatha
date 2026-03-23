"""Lambda API Gateway Bridge for Project Chitrakatha.

Architecture:
    API Gateway → Lambda → (pgvector RDS + Bedrock Qwen3 Next 80B A3B)

Flow per request:
    1. Parse and validate incoming query (Pydantic).
    2. Detect language (Devanagari → "hi", otherwise "en").
    3. Fetch RDS credentials from Secrets Manager.
    4. Embed query via Bedrock Titan Embed v2.
    5. Retrieve top-5 chunks from pgvector RDS (cosine similarity).
    6. Build RAFT-style prompt with retrieved context.
    7. Generate answer via Bedrock Qwen3 Next 80B A3B.
    8. Return answer + sources + language.

Lambda env vars (injected by Terraform lambda.tf):
    DB_SECRET_ARN          — Secrets Manager ARN for RDS credentials JSON
    BEDROCK_QWEN3_MODEL_ID — Qwen3 model ID for generation
    BEDROCK_EMBED_MODEL_ID — Titan Embed v2 model ID for query embedding
"""

import json
import logging
import os
import re

import boto3
import psycopg2
from botocore.exceptions import ClientError
from pydantic import BaseModel, ValidationError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_CORS_HEADERS = {
    "Content-Type": "application/json",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type",
    "Access-Control-Allow-Methods": "POST, OPTIONS",
}

DB_SECRET_ARN = os.environ.get("DB_SECRET_ARN", "")
QWEN3_MODEL_ID = os.environ.get("BEDROCK_QWEN3_MODEL_ID", "qwen.qwen3-next-80b-a3b")
EMBED_MODEL_ID = os.environ.get("BEDROCK_EMBED_MODEL_ID", "amazon.titan-embed-text-v2:0")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
_TOP_K = 5

# Global clients reused across warm invocations (cold-start mitigation).
bedrock = boto3.client("bedrock-runtime", region_name=AWS_REGION)
sm_client = boto3.client("secretsmanager", region_name=AWS_REGION)

_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class QueryRequest(BaseModel):
    """API Gateway incoming request schema."""

    query: str


def _get_db_creds() -> dict:
    """Fetch RDS credentials JSON from Secrets Manager."""
    resp = sm_client.get_secret_value(SecretId=DB_SECRET_ARN)
    return json.loads(resp["SecretString"])  # type: ignore[no-any-return]


def _embed(text: str) -> list[float]:
    """Embed text via Bedrock Titan Embed v2, returning a 1024-dim vector."""
    resp = bedrock.invoke_model(
        modelId=EMBED_MODEL_ID,
        contentType="application/json",
        accept="application/json",
        body=json.dumps({"inputText": text}),
    )
    return json.loads(resp["body"].read())["embedding"]  # type: ignore[no-any-return]


def _retrieve(embedding: list[float], creds: dict) -> list[dict[str, str]]:
    """Query pgvector for the top-k most similar chunks (cosine distance)."""
    conn = psycopg2.connect(
        host=creds["host"],
        port=int(creds["port"]),
        dbname=creds["dbname"],
        user=creds["username"],
        password=creds["password"],
        connect_timeout=5,
        sslmode="require",
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT chunk_text, source_document
                FROM embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                (str(embedding), _TOP_K),
            )
            rows = cur.fetchall()
        return [{"text": r[0], "source_document": r[1]} for r in rows]
    finally:
        conn.close()


def _generate(query: str, chunks: list[dict[str, str]]) -> str:
    """Generate an answer via Bedrock Qwen3 using retrieved chunks as context."""
    doc_blocks = "\n\n".join(
        f"[Document {i + 1}]: {c['text']}"
        for i, c in enumerate(chunks)
    )
    prompt = (
        f"You are given the following documents:\n{doc_blocks}\n\n"
        f"Question: {query}\n\n"
        "Answer using ONLY the relevant documents above. "
        "If none are relevant, say so.\nAnswer:"
    )
    resp = bedrock.converse(
        modelId=QWEN3_MODEL_ID,
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 512, "temperature": 0.1},
    )
    return resp["output"]["message"]["content"][0]["text"].strip()  # type: ignore[no-any-return]


def handler(event: dict, context: object) -> dict:
    """AWS Lambda entry point."""
    logger.info("Received event keys: %s", list(event.keys()))

    body_str = event.get("body", "{}") if "body" in event else json.dumps(event)

    try:
        req = QueryRequest(**json.loads(body_str))
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Invalid request: %s", exc)
        return {
            "statusCode": 400,
            "headers": _CORS_HEADERS,
            "body": json.dumps({"error": "Invalid request payload", "details": str(exc)}),
        }

    lang = "hi" if _DEVANAGARI_RE.search(req.query) else "en"

    try:
        creds = _get_db_creds()
        embedding = _embed(req.query)
        chunks = _retrieve(embedding, creds)

        if not chunks:
            return {
                "statusCode": 200,
                "headers": _CORS_HEADERS,
                "body": json.dumps(
                    {
                        "answer": "I don't have enough context in the knowledge base to answer that.",
                        "sources": [],
                        "language": lang,
                    },
                    ensure_ascii=False,
                ),
            }

        answer = _generate(req.query, chunks)
        sources = sorted({c["source_document"] for c in chunks})

        return {
            "statusCode": 200,
            "headers": _CORS_HEADERS,
            "body": json.dumps(
                {"answer": answer, "sources": sources, "language": lang},
                ensure_ascii=False,
            ),
        }

    except Exception as exc:
        logger.error("RAG pipeline failed: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "headers": _CORS_HEADERS,
            "body": json.dumps({"error": "Internal server error"}),
        }
