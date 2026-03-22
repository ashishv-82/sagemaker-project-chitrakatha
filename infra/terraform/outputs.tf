###############################################################################
# Project Chitrakatha — Terraform Outputs
#
# Why: All downstream Python scripts (pipeline.py, deploy_endpoint.py,
#      inference.py, lambda/handler.py) derive their config exclusively from
#      environment variables that are populated from these Terraform outputs.
#      No ARN or bucket name is ever hardcoded in Python or GitHub Actions.
#
# CI/CD pattern:
#   `terraform output -json > tf_outputs.json`
#   GitHub Actions reads this file and injects values as env vars before
#   running Python scripts or the SageMaker Pipeline.
###############################################################################

output "sagemaker_role_arn" {
  description = "ARN of the SageMaker execution role. Passed to all pipeline steps and the endpoint deploy script."
  value       = aws_iam_role.sagemaker_execution.arn
}

output "lambda_role_arn" {
  description = "ARN of the Lambda execution role. Used by lambda.tf (Phase 4) to wire the API Gateway bridge."
  value       = aws_iam_role.lambda_execution.arn
}

output "kms_key_arn" {
  description = "ARN of the Customer Managed KMS Key. Used wherever boto3 calls require explicit encryption key configuration."
  value       = aws_kms_key.chitrakatha.arn
}

output "kms_key_alias" {
  description = "Human-readable alias for the CMK. Use in Terraform resource references within the same root module."
  value       = aws_kms_alias.chitrakatha.name
}

output "s3_bronze_bucket" {
  description = "Name of the Bronze S3 bucket (raw ingest). Use as `BRONZE_BUCKET` env var in ingestion scripts."
  value       = aws_s3_bucket.bronze.id
}

output "s3_bronze_bucket_arn" {
  description = "ARN of the Bronze S3 bucket."
  value       = aws_s3_bucket.bronze.arn
}

output "s3_silver_bucket" {
  description = "Name of the Silver S3 bucket (cleaned JSONL). Use as `SILVER_BUCKET` env var."
  value       = aws_s3_bucket.silver.id
}

output "s3_silver_bucket_arn" {
  description = "ARN of the Silver S3 bucket."
  value       = aws_s3_bucket.silver.arn
}

output "s3_gold_bucket" {
  description = "Name of the Gold S3 bucket (training JSONL + checkpoints + evaluation results). Use as `GOLD_BUCKET` env var."
  value       = aws_s3_bucket.gold.id
}

output "s3_gold_bucket_arn" {
  description = "ARN of the Gold S3 bucket."
  value       = aws_s3_bucket.gold.arn
}

output "s3_vectors_bucket" {
  description = "Name of the S3 Vectors bucket. Use as `VECTORS_BUCKET` env var in faiss_writer.py and inference.py."
  value       = aws_s3_bucket.vectors.id
}

output "s3_vectors_bucket_arn" {
  description = "ARN of the S3 Vectors bucket."
  value       = aws_s3_bucket.vectors.arn
}

output "s3_faiss_index_prefix" {
  description = "Prefix of the FAISS index stored in S3. Injected into the serverless endpoint environment as `S3_FAISS_INDEX_PREFIX`."
  value       = local.s3_faiss_index_prefix
}

output "secret_arn" {
  description = "ARN of the Secrets Manager secret for the synthetic data API key. Use as `SECRET_ARN` env var in synthesize_training_pairs.py."
  value       = aws_secretsmanager_secret.synthetic_data_api_key.arn
}

output "secret_name" {
  description = "Name of the Secrets Manager secret. Use directly in boto3 get_secret_value calls."
  value       = aws_secretsmanager_secret.synthetic_data_api_key.name
}

output "cloudwatch_dashboard_url" {
  description = "Direct URL to the ChitrakathaMLOpsDashboard in the AWS Console."
  value       = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.chitrakatha.dashboard_name}"
}

output "api_gateway_invoke_url" {
  description = "The public HTTP URL of the RAG API. Use this to send POST requests containing {'query': '...'}."
  value       = "${aws_apigatewayv2_stage.default.invoke_url}v1/query"
}

output "lambda_function_arn" {
  description = "The ARN of the deployed Lambda bridge function."
  value       = aws_lambda_function.api_bridge.arn
}
output "github_actions_role_arn" {
  description = "ARN of the GitHub Actions OIDC role. Use this as AWS_ROLE_ARN in GitHub Secrets."
  value       = aws_iam_role.github_actions.arn
}

output "sagemaker_studio_domain_id" {
  description = "SageMaker Studio domain ID. Navigate to this URL to open Studio: https://{region}.console.aws.amazon.com/sagemaker/home?region={region}#/studio"
  value       = aws_sagemaker_domain.chitrakatha.id
}

output "sagemaker_studio_url" {
  description = "Direct link to open SageMaker Studio in the AWS console."
  value       = "https://${var.aws_region}.console.aws.amazon.com/sagemaker/home?region=${var.aws_region}#/studio/${aws_sagemaker_domain.chitrakatha.id}/open?profileName=${var.studio_user_profile_name}"
}
