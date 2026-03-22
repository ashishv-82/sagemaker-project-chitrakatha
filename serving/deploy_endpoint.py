"""SageMaker Real-time Inference Endpoint Deployment with Scale-to-Zero.

Why: Switched from Serverless Inference to a real-time GPU endpoint.
     SageMaker Serverless is CPU-only (6 GB max RAM), which cannot run the
     fine-tuned Llama 3.2 3B model even in bfloat16 (~6 GB weights alone).
     A real-time ml.g4dn.xlarge endpoint with Application Auto Scaling
     (MinCapacity=0) provides GPU-backed inference with true scale-to-zero:
     the instance terminates after SCALE_IN_COOLDOWN seconds of idle traffic,
     and SageMaker boots a new one (~3-5 min cold start) when demand returns.

Endpoint configuration:
    Instance type:   ml.g4dn.xlarge  (16 GB VRAM, NVIDIA T4, ~$0.74/hr)
    Auto Scaling:    MinCapacity=0, MaxCapacity=2
    Scale-out:       SageMakerVariantInvocationsPerInstance > 5 → add instance
    Scale-in:        600 s cooldown after invocations drop (scale to 0)

Cold-start behaviour:
    When the endpoint is at 0 instances, invoke_endpoint raises a ModelError
    (HTTP 424). The Lambda bridge (handler.py) returns HTTP 503 with a
    Retry-After: 300 header so callers know to retry after ~5 minutes.

IAM requirements (GitHub Actions role must have):
    - sagemaker:CreateEndpoint / UpdateEndpoint / DescribeEndpoint
    - application-autoscaling:RegisterScalableTarget
    - application-autoscaling:PutScalingPolicy
    These are in addition to the existing sagemaker:* grants in iam.tf.
"""

from __future__ import annotations

import logging
import os
import sys
import time

import boto3
from sagemaker import ModelPackage, Session

from chitrakatha.config import Settings

logger = logging.getLogger(__name__)

# Matches the model package group name used in pipeline.py RegisterModel step.
MODEL_PACKAGE_GROUP = "ChitrakathaModelPackageGroup"
ENDPOINT_NAME = "chitrakatha-rag-endpoint"

# Real-time instance: ml.g4dn.xlarge — 16 GB VRAM, NVIDIA T4.
# 3B model in 4-bit NF4 uses ~2.5 GB VRAM; plenty of headroom.
INSTANCE_TYPE = "ml.g4dn.xlarge"
INITIAL_INSTANCE_COUNT = 1

# Application Auto Scaling — scale-to-zero configuration.
MAX_INSTANCES = 2
SCALE_IN_COOLDOWN = 600    # seconds of idle before scaling in (to 0)
SCALE_OUT_COOLDOWN = 60    # seconds before scaling out on demand spike
TARGET_INVOCATIONS = 5.0   # target invocations/instance/min for tracking policy


def _get_latest_approved_model(sm_client: boto3.client) -> str:
    """Return the ARN of the newest Approved model package in the registry."""
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
    """Poll until the endpoint reaches InService (or Failed)."""
    logger.info("Waiting for endpoint '%s' to become InService...", endpoint_name)
    while True:
        resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = resp["EndpointStatus"]
        if status == "InService":
            logger.info("Endpoint is InService.")
            return
        if status == "Failed":
            raise RuntimeError(f"Endpoint creation failed: {resp.get('FailureReason', 'Unknown')}")
        logger.info("  current status: %s — waiting 30 s", status)
        time.sleep(30)


def _configure_autoscaling(region: str) -> None:
    """Register the endpoint with App Auto Scaling and attach a tracking policy.

    MinCapacity=0 enables scale-to-zero: SageMaker will terminate the instance
    after SCALE_IN_COOLDOWN seconds of idle invocations, and automatically
    boot a new one when the next request arrives (~3-5 min cold start on GPU).

    The target tracking policy maintains ~5 invocations per instance per minute.
    When traffic drops below this for SCALE_IN_COOLDOWN seconds, the instance
    count decreases toward 0.
    """
    resource_id = f"endpoint/{ENDPOINT_NAME}/variant/AllTraffic"
    aas = boto3.client("application-autoscaling", region_name=region)

    # Register the endpoint variant as a scalable target.
    aas.register_scalable_target(
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        MinCapacity=0,
        MaxCapacity=MAX_INSTANCES,
    )
    logger.info("Registered scalable target: min=0, max=%d instances.", MAX_INSTANCES)

    # Target tracking: scale out when invocations/instance exceed the target,
    # scale in (toward 0) when they drop below it for SCALE_IN_COOLDOWN seconds.
    aas.put_scaling_policy(
        PolicyName=f"{ENDPOINT_NAME}-target-tracking",
        ServiceNamespace="sagemaker",
        ResourceId=resource_id,
        ScalableDimension="sagemaker:variant:DesiredInstanceCount",
        PolicyType="TargetTrackingScaling",
        TargetTrackingScalingPolicyConfiguration={
            "TargetValue": TARGET_INVOCATIONS,
            "PredefinedMetricSpecification": {
                "PredefinedMetricType": "SageMakerVariantInvocationsPerInstance",
            },
            "ScaleInCooldown": SCALE_IN_COOLDOWN,
            "ScaleOutCooldown": SCALE_OUT_COOLDOWN,
        },
    )
    logger.info(
        "Auto-scaling configured: target=%.0f invoc/instance/min, "
        "scale-in cooldown=%d s.",
        TARGET_INVOCATIONS,
        SCALE_IN_COOLDOWN,
    )


def deploy() -> None:
    """Deploy or update the Chitrakatha real-time endpoint, then configure auto-scaling."""
    settings = Settings()
    sm_client = boto3.client("sagemaker", region_name=settings.aws_region)
    session = Session(sagemaker_client=sm_client)

    logger.info("Locating approved model in registry...")
    model_arn = _get_latest_approved_model(sm_client)
    logger.info("Deploying model package: %s", model_arn)

    model = ModelPackage(
        role=settings.sagemaker_role_arn,
        model_package_arn=model_arn,
        sagemaker_session=session,
        # Inject RAG env vars so inference.py can reach S3 Vectors.
        env={
            "S3_VECTORS_BUCKET": settings.s3_vectors_bucket,
            "S3_FAISS_INDEX_PREFIX": settings.s3_faiss_index_prefix,
            "AWS_REGION": settings.aws_region,
        },
    )

    update_endpoint = False
    try:
        sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
        logger.info("Endpoint '%s' exists — will UPDATE.", ENDPOINT_NAME)
        update_endpoint = True
    except sm_client.exceptions.ClientError as exc:
        if "Could not find endpoint" not in str(exc):
            raise

    logger.info(
        "%s endpoint '%s' on %s (real-time, scale-to-zero).",
        "Updating" if update_endpoint else "Creating",
        ENDPOINT_NAME,
        INSTANCE_TYPE,
    )

    model.deploy(
        endpoint_name=ENDPOINT_NAME,
        instance_type=INSTANCE_TYPE,
        initial_instance_count=INITIAL_INSTANCE_COUNT,
        update_endpoint=update_endpoint,
        wait=False,  # We poll ourselves for better log visibility.
    )

    _wait_for_endpoint(sm_client, ENDPOINT_NAME)
    _configure_autoscaling(settings.aws_region)
    logger.info(
        "Deployment complete. Endpoint will scale to 0 after %d s of idle traffic.",
        SCALE_IN_COOLDOWN,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        deploy()
    except Exception as exc:
        logger.error("Deployment failed: %s", exc)
        sys.exit(1)
