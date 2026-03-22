"""SageMaker Serverless Endpoint Deployment.

Why: Deploys the latest approved model from the Chitrakatha Model Registry
     to a SageMaker Serverless Endpoint. Serverless is ideal for this RAG API
     because it scales automatically and charges only for compute time used
     (scale-to-zero when idle) — perfect for unpredictable consumer API traffic.

Capabilities:
    - Finds the latest 'Approved' model package in the group.
    - Creates a serverless endpoint configuration (6144 MB RAM, max concurrency 5).
    - Injects ``S3_FAISS_INDEX_PREFIX`` and ``S3_VECTORS_BUCKET`` so the
      inference container can query the RAG knowledge base.
    - Idempotent: Updates the endpoint if it exists, creates if it doesn't.

Constraints:
    - Waits synchronously for the endpoint to become InService.
    - Role ARN and AWS Region are pulled from environment variables.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import boto3
from sagemaker import Session
from sagemaker.serverless import ServerlessInferenceConfig

from chitrakatha.config import Settings

logger = logging.getLogger(__name__)

# Constants matching Phase 3 model registration.
MODEL_PACKAGE_GROUP = "ChitrakathaModelPackageGroup"
ENDPOINT_NAME = "chitrakatha-rag-serverless"
MEMORY_SIZE_MB = 6144
MAX_CONCURRENCY = 5


def _get_latest_approved_model(sm_client: boto3.client) -> str:
    """Find the ARN of the newest Approved model in the registry."""
    paginator = sm_client.get_paginator("list_model_packages")
    for page in paginator.paginate(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelApprovalStatus="Approved",
        SortBy="CreationTime",
        SortOrder="Descending",
    ):
        for pkg in page.get("ModelPackageSummaryList", []):
            return pkg["ModelPackageArn"]
            
    raise RuntimeError(f"No 'Approved' models found in group: {MODEL_PACKAGE_GROUP}")


def _wait_for_endpoint(sm_client: boto3.client, endpoint_name: str) -> None:
    """Poll endpoint status until it reaches InService (or Fails)."""
    logger.info("Waiting for endpoint '%s' to become InService...", endpoint_name)
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        if status == "InService":
            logger.info("Endpoint successfully deployed and InService!")
            return
        if status == "Failed":
            reason = resp.get("FailureReason", "Unknown")
            raise RuntimeError(f"Endpoint creation failed: {reason}")
            
        logger.info("  current status: %s", status)
        time.sleep(30)


def deploy() -> None:
    """Deploy or update the Chitrakatha Serverless Endpoint."""
    settings = Settings()
    sm_client = boto3.client("sagemaker", region_name=settings.aws_region)
    session = Session(sagemaker_client=sm_client)

    logger.info("Locating approved model in registry...")
    model_arn = _get_latest_approved_model(sm_client)
    logger.info("Deploying model package: %s", model_arn)

    # Creating a sagemaker.model.ModelPackage to deploy from the registry ARN.
    from sagemaker import ModelPackage  # noqa: PLC0415
    model = ModelPackage(
        role=settings.sagemaker_role_arn,
        model_package_arn=model_arn,
        sagemaker_session=session,
        # Inject RAG environment variables so inference.py can contact S3 Vectors.
        env={
            "S3_VECTORS_BUCKET": settings.s3_vectors_bucket,
            "S3_FAISS_INDEX_PREFIX": settings.s3_faiss_index_prefix,
            "AWS_REGION": settings.aws_region,
        },
    )

    serverless_config = ServerlessInferenceConfig(
        memory_size_in_mb=MEMORY_SIZE_MB,
        max_concurrency=MAX_CONCURRENCY,
    )

    # Check if the endpoint already exists.
    update_endpoint = False
    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info("Endpoint %s exists. Will UPDATE it.", ENDPOINT_NAME)
        update_endpoint = True
    except sm_client.exceptions.ClientError as exc:
        if "Could not find endpoint" not in str(exc):
            raise

    # Trigger deployment. This usually takes 3-5 minutes.
    logger.info(
        "Initiating %s deployment to %s (Serverless: %dMB, max_concurrency: %d)",
        "UPDATE" if update_endpoint else "CREATE",
        ENDPOINT_NAME,
        MEMORY_SIZE_MB,
        MAX_CONCURRENCY,
    )
    
    model.deploy(
        endpoint_name=ENDPOINT_NAME,
        serverless_inference_config=serverless_config,
        update_endpoint=update_endpoint,
        wait=False, # We implement our own waiting so we get better logs.
    )
    
    _wait_for_endpoint(sm_client, ENDPOINT_NAME)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        deploy()
    except Exception as e:
        logger.error("Deployment failed: %s", e)
        sys.exit(1)
