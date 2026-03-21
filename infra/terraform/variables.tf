###############################################################################
# Project Chitrakatha — Input Variables
#
# Why: Centralises all tuneable parameters so there are zero hardcoded values
#      in resource files. CI/CD overrides these via tfvars or TF_VAR_* env vars.
#
# Constraints:
#   - Never add secrets, ARNs, or account IDs here; use data sources instead.
#   - Validation blocks prevent invalid deployments early (before `apply`).
###############################################################################

variable "aws_region" {
  description = "AWS region where all Chitrakatha resources are deployed."
  type        = string
  default     = "ap-southeast-2"
}

variable "project_name" {
  description = <<-EOT
    Short name used as a prefix for all resource identifiers (S3 buckets,
    IAM roles, KMS aliases, etc.). Must be lowercase alphanumeric + hyphens.
  EOT
  type    = string
  default = "chitrakatha"

  validation {
    condition     = can(regex("^[a-z0-9-]+$", var.project_name))
    error_message = "project_name must be lowercase alphanumeric characters and hyphens only."
  }
}

variable "environment" {
  description = "Deployment tier. Controls resource naming and lifecycle strictness."
  type        = string
  default     = "dev"

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "kms_deletion_window_days" {
  description = <<-EOT
    Number of days before a scheduled KMS key deletion takes effect.
    AWS minimum is 7; use 30 for production to allow recovery.
  EOT
  type    = number
  default = 30

  validation {
    condition     = var.kms_deletion_window_days >= 7 && var.kms_deletion_window_days <= 30
    error_message = "kms_deletion_window_days must be between 7 and 30."
  }
}

variable "alarm_sns_topic_arn" {
  description = <<-EOT
    ARN of the SNS topic that receives CloudWatch alarm notifications (cold-start,
    error rate, etc.). Leave empty in dev to suppress alerts.
  EOT
  type    = string
  default = ""
}

variable "s3_vector_dimension" {
  description = <<-EOT
    Dimensionality of the embedding vectors stored in the S3 Vectors index.
    Must match the output size of the embedding model.
    Titan Embed Text v2 produces 1536-dimensional vectors.
  EOT
  type    = number
  default = 1536
}

variable "s3_vector_metric" {
  description = "Distance metric used for nearest-neighbour search in the S3 Vectors index."
  type        = string
  default     = "cosine"

  validation {
    condition     = contains(["cosine", "euclidean", "dot_product"], var.s3_vector_metric)
    error_message = "s3_vector_metric must be one of: cosine, euclidean, dot_product."
  }
}

variable "noncurrent_version_expiry_days" {
  description = "Days after which non-current S3 object versions are permanently deleted."
  type        = number
  default     = 365
}

variable "noncurrent_version_ia_transition_days" {
  description = "Days after which non-current S3 object versions are moved to STANDARD_IA."
  type        = number
  default     = 30
}
