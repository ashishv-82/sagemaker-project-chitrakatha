"""
Configuration management using pydantic v2 BaseSettings.
Outputs from Terraform (ARNs, bucket names) should be passed via environment variables.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Core settings for Project Chitrakatha."""
    
    # AWS Region
    aws_region: str = Field(default="us-east-1", alias="AWS_REGION")
    
    # Storage
    s3_bronze_bucket: str = Field(..., alias="S3_BRONZE_BUCKET")
    s3_silver_bucket: str = Field(..., alias="S3_SILVER_BUCKET")
    s3_gold_bucket: str = Field(..., alias="S3_GOLD_BUCKET")
    s3_vectors_bucket: str = Field(..., alias="S3_VECTORS_BUCKET")
    
    # Infrastructure ARNs
    kms_key_arn: str = Field(..., alias="KMS_KEY_ARN")
    sagemaker_role_arn: str = Field(..., alias="SAGEMAKER_ROLE_ARN")
    
    # Bedrock
    bedrock_embed_model_id: str = Field(
        default="amazon.titan-embed-text-v2:0", 
        alias="BEDROCK_EMBED_MODEL_ID"
    )
    bedrock_teacher_model_id: str = Field(
        default="anthropic.claude-3-5-sonnet-20241022-v2:0", 
        alias="BEDROCK_TEACHER_MODEL_ID"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Global settings instance
config = Settings()  # type: ignore
