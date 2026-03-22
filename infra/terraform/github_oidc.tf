###############################################################################
# Project Chitrakatha — GitHub OIDC Authentication (Phase 6)
#
# Why: Using OIDC is the security gold standard for GitHub Actions. It eliminates
#      the need for long-lived IAM Access Keys. GitHub provides a JWT, and AWS
#      exchanges it for short-lived credentials via an IAM Role with OIDC trust.
###############################################################################

# 1. GitHub OIDC Provider (Common across all repositories in the account)
# NOTE: If you already have this provider, you can remove this block and use a data-source.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1", "1c58a3a8518e8759bf075b76b750d4f2df264fcd"]
}

# 2. Trust Policy for the GitHub Role
data "aws_iam_policy_document" "github_actions_assume_role" {
  statement {
    actions = ["sts:AssumeRoleWithWebIdentity"]
    effect  = "Allow"

    principals {
      type        = "Federated"
      identifiers = [aws_iam_openid_connect_provider.github.arn]
    }

    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      # Only allow your specific repository to assume this role.
      values = ["repo:ashishv-82/sagemaker-project-chitrakatha:*"]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }
  }
}

# 3. The GitHub Actions IAM Role
resource "aws_iam_role" "github_actions" {
  name               = "${var.project_name}-github-actions"
  assume_role_policy = data.aws_iam_policy_document.github_actions_assume_role.json

  tags = {
    Purpose = "GitHub Actions CI/CD/CT identity"
  }
}

# 4. Permissions for GitHub Actions
# This role needs to:
#   - Trigger SageMaker Pipelines
#   - Manage Terraform State in S3 (if you use remote backend)
#   - Upload scripts to S3 for processing jobs
data "aws_iam_policy_document" "github_actions_permissions" {
  statement {
    sid    = "SageMakerPipelineTrigger"
    effect = "Allow"
    actions = [
      "sagemaker:StartPipelineExecution",
      "sagemaker:DescribePipelineExecution",
      "sagemaker:ListPipelineExecutionSteps",
      "sagemaker:DescribePipeline",
      "sagemaker:UpsertPipeline",
      "sagemaker:CreatePipeline",
      "sagemaker:UpdatePipeline",
      "sagemaker:AddTags",
      "sagemaker:ListTags"
    ]
    resources = [
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:pipeline/${var.project_name}-*"
    ]
  }

  statement {
    sid    = "S3UploadScriptsAndData"
    effect = "Allow"
    actions = [
      "s3:PutObject",
      "s3:GetObject",
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.bronze.arn,
      "${aws_s3_bucket.bronze.arn}/*",
      aws_s3_bucket.silver.arn,
      "${aws_s3_bucket.silver.arn}/*",
      aws_s3_bucket.gold.arn,
      "${aws_s3_bucket.gold.arn}/*"
    ]
  }

  statement {
    sid    = "TerraformStateBackend"
    effect = "Allow"
    actions = [
      "s3:ListBucket",
      "s3:GetObject",
      "s3:PutObject",
      "dynamodb:GetItem",
      "dynamodb:PutItem",
      "dynamodb:DeleteItem"
    ]
    resources = [
      "arn:aws:s3:::project-chitrakatha-tf-state-*",
      "arn:aws:s3:::project-chitrakatha-tf-state-*/*",
      "arn:aws:dynamodb:${var.aws_region}:${local.account_id}:table/project-chitrakatha-tf-lock"
    ]
  }

  # Allow GitHub Actions to "Pass" the SageMaker role to the service.
  statement {
    sid       = "IAMPassRoleToSageMaker"
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_execution.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }

  # deploy_endpoint.py: find approved model, create model/config, deploy endpoint.
  # Endpoint + config resources use a wildcard suffix because SageMaker
  # auto-generates a timestamp suffix on the config name at deploy time.
  statement {
    sid    = "SageMakerEndpointDeploy"
    effect = "Allow"
    actions = [
      "sagemaker:ListModelPackages",
      "sagemaker:DescribeModelPackage",
      "sagemaker:CreateModel",
      "sagemaker:DeleteModel",
      "sagemaker:DescribeModel",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:CreateEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:DescribeEndpoint",
    ]
    resources = [
      # Endpoint (exact name used by deploy_endpoint.py).
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint/${var.project_name}-rag-endpoint",
      # Endpoint configs get a timestamp suffix — prefix-match covers them.
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:endpoint-config/${var.project_name}-*",
      # Model name is auto-generated by SageMaker SDK — scoped to account only.
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:model/*",
      # Model registry — package group and all packages within it.
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:model-package-group/ChitrakathaModelPackageGroup",
      "arn:aws:sagemaker:${var.aws_region}:${local.account_id}:model-package/ChitrakathaModelPackageGroup/*",
    ]
  }

  # deploy_endpoint.py: register endpoint with App Auto Scaling and attach
  # target tracking policy (MinCapacity=0 → scale-to-zero).
  # Application Auto Scaling does not support resource-level restrictions
  # for RegisterScalableTarget / PutScalingPolicy, so Resource = "*" is required.
  statement {
    sid    = "AppAutoScalingEndpoint"
    effect = "Allow"
    actions = [
      "application-autoscaling:RegisterScalableTarget",
      "application-autoscaling:PutScalingPolicy",
      "application-autoscaling:DescribeScalableTargets",
      "application-autoscaling:DescribeScalingPolicies",
    ]
    resources = ["*"]
  }

  # First-time setup: App Auto Scaling needs to create its SageMaker
  # service-linked role (AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint)
  # if it does not exist in the account. Scoped to that exact role path.
  statement {
    sid     = "AppAutoScalingServiceLinkedRole"
    effect  = "Allow"
    actions = ["iam:CreateServiceLinkedRole"]
    resources = [
      "arn:aws:iam::*:role/aws-service-role/sagemaker.application-autoscaling.amazonaws.com/AWSServiceRoleForApplicationAutoScaling_SageMakerEndpoint"
    ]
    condition {
      test     = "StringLike"
      variable = "iam:AWSServiceName"
      values   = ["sagemaker.application-autoscaling.amazonaws.com"]
    }
  }

  # _sync_chitrakatha_to_s3() uploads to KMS-encrypted Silver bucket; SageMaker
  # pipeline.upsert() uploads source_dir tarballs to KMS-encrypted Gold bucket.
  # Both require GenerateDataKey (encrypt) and Decrypt (read-back verification).
  statement {
    sid    = "KMSEncryptDecryptForS3Uploads"
    effect = "Allow"
    actions = [
      "kms:GenerateDataKey",
      "kms:Decrypt",
      "kms:DescribeKey",
    ]
    resources = [aws_kms_key.chitrakatha.arn]
  }

  # pipeline.py calls model_uris.retrieve() to fetch JumpStart model metadata.
  # The SageMaker SDK reads this from an AWS-managed public S3 bucket
  # (jumpstart-cache-prod-{region}). Without this grant the pipeline definition
  # step fails with AccessDenied even though the bucket is AWS-owned.
  statement {
    sid    = "JumpStartModelCacheRead"
    effect = "Allow"
    actions = [
      "s3:GetObject",
    ]
    resources = [
      "arn:aws:s3:::jumpstart-cache-prod-${var.aws_region}/*",
    ]
  }
}

resource "aws_iam_role_policy" "github_actions_policy" {
  name   = "github-actions-cis-permissions"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions_permissions.json
}
