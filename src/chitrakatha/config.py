"""Configuration management for Project Chitrakatha.

Why: All config (S3 bucket names, KMS key ARNs, SageMaker role ARN) is sourced
     from environment variables populated by ``terraform output`` in CI/CD.
     Using ``pydantic-settings`` BaseSettings provides automatic env var reading,
     type coercion, and validation — no hardcoded values, no manual os.environ
     lookups scattered across scripts.

Constraints:
    - All required fields (marked with ``...``) will raise ``ValidationError``
      at instantiation if the corresponding env var is not set. This ensures
      misconfigured deployments fail fast rather than silently.
    - ``global_settings`` is NOT instantiated at module level — doing so would
      crash test imports that don't have AWS env vars set. Callers must call
      ``Settings()`` explicitly (or use the factory function ``get_settings()``).
    - Optional fields have sensible defaults that work in dev without AWS.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed, validated runtime configuration for Project Chitrakatha.

    All fields are populated from environment variables (names matching the
    field alias). A ``.env`` file is also supported for local development.
    In production, env vars are injected by GitHub Actions from Terraform outputs.
    """

    # -- AWS ------------------------------------------------------------------
    aws_region: str = Field(default="ap-southeast-2", alias="AWS_REGION")

    # -- S3 buckets (required — names are Terraform outputs) ------------------
    s3_bronze_bucket: str = Field(..., alias="S3_BRONZE_BUCKET")
    s3_silver_bucket: str = Field(..., alias="S3_SILVER_BUCKET")
    s3_gold_bucket: str = Field(..., alias="S3_GOLD_BUCKET")
    s3_vectors_bucket: str = Field(..., alias="S3_VECTORS_BUCKET")

    # -- FAISS Index (required) ------------------------------------------
    s3_faiss_index_prefix: str = Field(..., alias="S3_FAISS_INDEX_PREFIX")

    # -- KMS & IAM (required) -------------------------------------------------
    kms_key_arn: str = Field(..., alias="KMS_KEY_ARN")
    sagemaker_role_arn: str = Field(..., alias="SAGEMAKER_ROLE_ARN")

    # -- Secrets Manager (required) -------------------------------------------
    secret_name: str = Field(
        default="chitrakatha/synthetic_data_api_key",
        alias="SECRET_NAME",
    )

    # -- Bedrock model IDs (optional — sensible defaults) --------------------
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0",
        alias="BEDROCK_EMBED_MODEL_ID",
    )
    bedrock_teacher_model_id: str = Field(
        default="ap.anthropic.claude-haiku-4-5-20251001-v1:0",
        alias="BEDROCK_TEACHER_MODEL_ID",
    )

    # -- SageMaker Experiments (optional — empty disables logging) -----------
    sagemaker_experiment_name: str = Field(
        default="chitrakatha-mlops",
        alias="SAGEMAKER_EXPERIMENT_NAME",
    )

    @field_validator("kms_key_arn")
    @classmethod
    def kms_arn_must_be_valid(cls, v: str) -> str:
        """Validate KMS ARN format — catches copy-paste errors early."""
        if v and not v.startswith("arn:aws:kms:"):
            raise ValueError(
                f"kms_key_arn must start with 'arn:aws:kms:', got: {v!r}. "
                "Use the ARN from `terraform output kms_key_arn`."
            )
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Devanagari characters in .env values are preserved as UTF-8.
        extra="ignore",
        populate_by_name=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` instance.

    Using ``lru_cache`` ensures the environment is read once per process —
    avoids repeated file I/O for ``.env`` on every function call.
    Tests that need custom settings should call ``Settings()`` directly
    with the desired env vars set, or use ``get_settings.cache_clear()``
    between test cases.

    Returns:
        Validated ``Settings`` instance.
    """
    return Settings()  # type: ignore[call-arg]

