###############################################################################
# Project Chitrakatha — IAM Roles & Policies
#
# Why: The SageMaker execution role is the single identity used by all pipeline
#      steps (Processing, Training, Model, Endpoint). Scoping its permissions
#      to exactly the 4 named S3 buckets, one KMS key, and specific Bedrock
#      model ARNs enforces the Principle of Least Privilege — no wildcard
#      resources anywhere.
#
# Constraints:
#   - No `Resource: "*"` on data-plane actions (S3, KMS, Bedrock, Secrets).
#   - SageMaker control-plane actions (Create*/Describe*) are scoped to the
#     project prefix where possible.
#   - This role ARN is exported in outputs.tf and injected into pipeline.py
#     and deploy_endpoint.py via environment variables — never hardcoded.
###############################################################################

###############################################################################
# SageMaker Execution Role
###############################################################################

data "aws_iam_policy_document" "sagemaker_assume_role" {
  # Only the SageMaker service can assume this role.
  statement {
    sid     = "AllowSageMakerAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_execution" {
  name               = "${var.project_name}-sagemaker-execution"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json

  tags = {
    Purpose = "SageMaker pipeline execution - all processing training and endpoint steps"
  }
}

###############################################################################
# Inline Policy: S3 Access (scoped to the 4 project buckets only)
###############################################################################

data "aws_iam_policy_document" "sagemaker_s3" {
  statement {
    sid    = "S3ReadWriteProjectBuckets"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = [
      aws_s3_bucket.bronze.arn,
      "${aws_s3_bucket.bronze.arn}/*",
      aws_s3_bucket.silver.arn,
      "${aws_s3_bucket.silver.arn}/*",
      aws_s3_bucket.gold.arn,
      "${aws_s3_bucket.gold.arn}/*",
      aws_s3_bucket.vectors.arn,
      "${aws_s3_bucket.vectors.arn}/*",
    ]
  }
}

resource "aws_iam_role_policy" "sagemaker_s3" {
  name   = "s3-project-buckets"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_s3.json
}

###############################################################################
# Inline Policy: KMS (scoped to the project CMK only)
###############################################################################

data "aws_iam_policy_document" "sagemaker_kms" {
  statement {
    sid    = "KMSDecryptGenerateProjectKey"
    effect = "Allow"
    actions = [
      "kms:GenerateDataKey",
      "kms:Decrypt",
      "kms:DescribeKey"
    ]
    resources = [aws_kms_key.chitrakatha.arn]
  }
}

resource "aws_iam_role_policy" "sagemaker_kms" {
  name   = "kms-project-cmk"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_kms.json
}

###############################################################################
# Inline Policy: Bedrock (scoped to Titan Embed v2 and Claude 3.5 Sonnet ARNs)
###############################################################################

data "aws_iam_policy_document" "sagemaker_bedrock" {
  statement {
    sid    = "BedrockInvokeProjectModels"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream"
    ]
    resources = [
      # Titan Embed Text v2 — used for corpus embedding (Flow A) and query embedding.
      "arn:aws:bedrock:${var.aws_region}::foundation-model/amazon.titan-embed-text-v2:0",
      # Claude 3.5 Sonnet — used for RAFT training pair synthesis (Flow B).
      "arn:aws:bedrock:${var.aws_region}::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
    ]
  }
}

resource "aws_iam_role_policy" "sagemaker_bedrock" {
  name   = "bedrock-invoke-project-models"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_bedrock.json
}

###############################################################################
# Inline Policy: Secrets Manager (scoped to chitrakatha/* prefix)
###############################################################################

data "aws_iam_policy_document" "sagemaker_secrets" {
  statement {
    sid    = "SecretsManagerReadProjectSecrets"
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
      "secretsmanager:DescribeSecret"
    ]
    resources = [
      "arn:aws:secretsmanager:${var.aws_region}:${local.account_id}:secret:${var.project_name}/*"
    ]
  }
}

resource "aws_iam_role_policy" "sagemaker_secrets" {
  name   = "secretsmanager-project-secrets"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_secrets.json
}

###############################################################################
# Inline Policy: CloudWatch Logs (pipeline step log groups)
###############################################################################

data "aws_iam_policy_document" "sagemaker_logs" {
  statement {
    sid    = "CloudWatchLogsWriteProjectGroups"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams"
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/sagemaker/*"
    ]
  }
}

resource "aws_iam_role_policy" "sagemaker_logs" {
  name   = "cloudwatch-logs-sagemaker"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_logs.json
}

###############################################################################
# Inline Policy: SageMaker Control Plane (pipeline steps need these)
###############################################################################

data "aws_iam_policy_document" "sagemaker_control_plane" {
  statement {
    sid    = "SageMakerControlPlaneProjectScope"
    effect = "Allow"
    actions = [
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
      "sagemaker:StopProcessingJob",
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:StopTrainingJob",
      "sagemaker:CreateModel",
      "sagemaker:DescribeModel",
      "sagemaker:DeleteModel",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:CreateEndpoint",
      "sagemaker:DescribeEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:InvokeEndpoint",
      "sagemaker:CreateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:ListModelPackages",
      "sagemaker:CreateExperiment",
      "sagemaker:DescribeExperiment",
      "sagemaker:CreateTrial",
      "sagemaker:DescribeTrial",
      "sagemaker:CreateTrialComponent",
      "sagemaker:AssociateTrialComponent",
      # Pipeline operations (used by pipeline.py to manage its own execution).
      "sagemaker:CreatePipeline",
      "sagemaker:DescribePipeline",
      "sagemaker:StartPipelineExecution",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps"
    ]
    # Scope to resources prefixed with the project name where AWS supports it.
    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:*/${var.project_name}-*"
    ]
  }
}

resource "aws_iam_role_policy" "sagemaker_control_plane" {
  name   = "sagemaker-control-plane-project-scope"
  role   = aws_iam_role.sagemaker_execution.id
  policy = data.aws_iam_policy_document.sagemaker_control_plane.json
}



###############################################################################
# Lambda Execution Role (for the API Gateway bridge in Phase 4)
###############################################################################

data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    sid     = "AllowLambdaAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_execution" {
  name               = "${var.project_name}-lambda-execution"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json

  tags = {
    Purpose = "Lambda bridge: API Gateway to SageMaker real-time endpoint"
  }
}

# Allow Lambda to invoke only the Chitrakatha real-time endpoint.
data "aws_iam_policy_document" "lambda_invoke_endpoint" {
  statement {
    sid    = "InvokeChitrakathaEndpointOnly"
    effect = "Allow"
    actions = [
      "sagemaker:InvokeEndpoint"
    ]
    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint/${var.project_name}-*"
    ]
  }

  statement {
    sid    = "CloudWatchLogsLambda"
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents"
    ]
    resources = [
      "arn:aws:logs:${var.aws_region}:${local.account_id}:log-group:/aws/lambda/${var.project_name}-*"
    ]
  }
}

resource "aws_iam_role_policy" "lambda_invoke_endpoint" {
  name   = "invoke-chitrakatha-endpoint"
  role   = aws_iam_role.lambda_execution.id
  policy = data.aws_iam_policy_document.lambda_invoke_endpoint.json
}
