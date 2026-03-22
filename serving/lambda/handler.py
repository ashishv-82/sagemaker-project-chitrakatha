"""Lambda API Gateway Bridge for Project Chitrakatha.

Why: Exposes the SageMaker real-time endpoint (which requires AWS SigV4 Auth)
     as a public HTTP API (via API Gateway). Handles JSON validation via
     Pydantic, translates to SageMaker's invoke_endpoint signature, and
     formats the response.

Features:
    - Pydantic v2 validation for incoming requests.
    - Global boto3 client for cold-start performance (reused across invocations).
    - Language detection: if query has Devanagari, tags response language as "hi"
      (the LLM automatically outputs Hindi due to RAFT training).
    - Structured JSON logging for CloudWatch.
    - Scale-to-zero awareness: when the endpoint has 0 instances (scaled down),
      SageMaker returns a ModelError. The handler returns HTTP 503 with a
      Retry-After: 300 header so clients know to retry after ~5 minutes while
      the GPU instance boots.
"""

import json
import logging
import os
import re

import boto3
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

# Global clients are preserved between Lambda invocations (cold-start mitigation).
# Endpoint name is injected by Terraform from `lambda.tf`.
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "chitrakatha-rag-endpoint")
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
sm_runtime = boto3.client("sagemaker-runtime", region_name=AWS_REGION)

# Devanagari block
_DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")


class QueryRequest(BaseModel):
    """API Gateway incoming request schema."""
    query: str


def handler(event: dict, context: object) -> dict:
    """AWS Lambda entry point."""
    logger.info("Received event: %s", json.dumps(event))

    # Support both direct Lambda invocation payload and API Gateway proxy payload
    body_str = event.get("body", "{}") if "body" in event else json.dumps(event)
    
    try:
        body_dict = json.loads(body_str)
        req = QueryRequest(**body_dict)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Invalid request: %s", exc)
        return {
            "statusCode": 400,
            "headers": _CORS_HEADERS,
            "body": json.dumps({"error": "Invalid request payload", "details": str(exc)}),
        }

    # Determine expected response language based on input query
    lang = "hi" if _DEVANAGARI_RE.search(req.query) else "en"

    # Invoke the SageMaker real-time endpoint.
    try:
        sm_payload = json.dumps({"query": req.query})
        logger.info("Invoking endpoint %s with payload len %d", ENDPOINT_NAME, len(sm_payload))

        response = sm_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType="application/json",
            Accept="application/json",
            Body=sm_payload,
        )

        result_str = response["Body"].read().decode("utf-8")
        result = json.loads(result_str)

        # Merge response from inference.py with the language tag.
        api_response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "language": lang,
        }

        return {
            "statusCode": 200,
            "headers": _CORS_HEADERS,
            "body": json.dumps(api_response, ensure_ascii=False),
        }

    except ClientError as exc:
        # boto3 raises ClientError (subclass: ModelError) when invoke_endpoint
        # gets a non-2xx from the container. When the endpoint is at 0 instances
        # (scaled to zero), SageMaker returns OriginalStatusCode 424.
        err_code = exc.response["Error"]["Code"]
        original_status = exc.response.get("OriginalStatusCode", 0)
        logger.warning("Endpoint ClientError [%s] original_status=%s", err_code, original_status)
        if err_code == "ModelError" and (
            original_status == 424 or "No instances available" in str(exc)
        ):
            return {
                "statusCode": 503,
                "headers": {**_CORS_HEADERS, "Retry-After": "300"},
                "body": json.dumps({
                    "error": "Endpoint is warming up — GPU instance is starting.",
                    "retry_after_seconds": 300,
                }),
            }
        return {
            "statusCode": 500,
            "headers": _CORS_HEADERS,
            "body": json.dumps({"error": "Model inference error", "details": str(exc)}),
        }

    except Exception as exc:
        logger.error("SageMaker invocation failed: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "headers": _CORS_HEADERS,
            "body": json.dumps({"error": "Internal server error"}),
        }
