###############################################################################
# Project Chitrakatha — KMS Customer Managed Key
#
# Why: All S3 buckets (Bronze/Silver/Gold/Vectors) must be encrypted with a
#      Customer Managed Key so we retain full control over key rotation,
#      access grants, and audit trails — not possible with AWS-managed keys.
#
# Circular-dependency note:
#   The KMS key policy deliberately uses AWS service principals (not the
#   SageMaker IAM role ARN) to avoid a Terraform dependency cycle between
#   kms.tf and iam.tf. The SageMaker execution role receives kms:GenerateDataKey
#   and kms:Decrypt permissions via its own inline policy in iam.tf.
#
# Constraints:
#   - Key rotation enabled (annual) — mandatory for fintech-grade posture.
#   - Deletion window >= 30 days in all environments.
#   - Alias follows `alias/chitrakatha-cmk` naming convention.
###############################################################################

resource "aws_kms_key" "chitrakatha" {
  description             = "CMK for Project Chitrakatha — encrypts all S3 data tiers and Secrets Manager secrets."
  deletion_window_in_days = var.kms_deletion_window_days
  enable_key_rotation     = true

  # Key policy: least-privilege.
  # - Root account retains emergency admin access (required by AWS).
  # - AWS service principals are allowed only the operations they need.
  # - No wildcard actions on service grant statements.
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [

      # Root account must always have full key admin rights to prevent lockout.
      {
        Sid    = "EnableRootAdminAccess"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
        Action   = "kms:*"
        Resource = "*"
      },

      # SageMaker service needs these to encrypt/decrypt training data and model artefacts.
      {
        Sid    = "AllowSageMakerService"
        Effect = "Allow"
        Principal = {
          Service = "sagemaker.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey",
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },

      # Bedrock needs these to call Titan Embed with KMS-encrypted input/output.
      {
        Sid    = "AllowBedrockService"
        Effect = "Allow"
        Principal = {
          Service = "bedrock.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey",
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      },



      # Secrets Manager needs to encrypt/decrypt secrets at rest.
      {
        Sid    = "AllowSecretsManagerService"
        Effect = "Allow"
        Principal = {
          Service = "secretsmanager.amazonaws.com"
        }
        Action = [
          "kms:GenerateDataKey",
          "kms:Decrypt",
          "kms:DescribeKey"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_kms_alias" "chitrakatha" {
  # Alias makes the key identifiable in CloudTrail logs and console.
  name          = "alias/${var.project_name}-cmk"
  target_key_id = aws_kms_key.chitrakatha.key_id
}
