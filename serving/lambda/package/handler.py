"""Lambda API Gateway Bridge for Project Chitrakatha.

Why: Exposes the SageMaker Serverless Endpoint (which requires AWS SigV4 Auth)
     as a public HTTP API (via API Gateway). Handles JSON validation via
     Pydantic, translates to SageMaker's invoke_endpoint signature, and
     formats the response.

Features:
    - Pydantic v2 validation for incoming requests.
    - Global boto3 client for cold-start performance (reused across invocations).
    - Language detection: if query has Devanagari, tags response language as "hi"
      (the LLM automatically outputs Hindi due to RAFT training).
    - Structured JSON logging for CloudWatch.
"""

import json
import logging
import os
import re

import boto3
from pydantic import BaseModel, ValidationError

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global clients are preserved between Lambda invocations (cold-start mitigation).
# Endpoint name is injected by Terraform from `lambda.tf`.
ENDPOINT_NAME = os.environ.get("SAGEMAKER_ENDPOINT_NAME", "chitrakatha-rag-serverless")
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
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Invalid request payload", "details": str(exc)}),
        }

    # Determine expected response language based on input query
    lang = "hi" if _DEVANAGARI_RE.search(req.query) else "en"

    # Invoke the SageMaker Serverless Endpoint
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
        
        # Merge response from inference.py with the language tag
        api_response = {
            "answer": result.get("answer", ""),
            "sources": result.get("sources", []),
            "language": lang,
        }

        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(api_response, ensure_ascii=False),
        }

    except Exception as exc:
        logger.error("SageMaker invocation failed: %s", exc, exc_info=True)
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps({"error": "Internal server error"}),
        }
