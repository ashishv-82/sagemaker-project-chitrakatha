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
      "sagemaker:UpsertPipeline"
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
}

resource "aws_iam_role_policy" "github_actions_policy" {
  name   = "github-actions-cis-permissions"
  role   = aws_iam_role.github_actions.id
  policy = data.aws_iam_policy_document.github_actions_permissions.json
}
