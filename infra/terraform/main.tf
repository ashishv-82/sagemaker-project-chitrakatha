###############################################################################
# Project Chitrakatha — Terraform Root Configuration
#
# Why: Pins the AWS provider to >= 5.90 which is the minimum version that
#      includes the aws_s3_vectors_index resource (2026 native S3 Vectors API).
#      Uses an S3 backend for remote state with DynamoDB locking — bootstrapped
#      separately via scripts/bootstrap_backend.sh before first `terraform init`.
#
# Constraints:
#   - Terraform >= 1.7 required for improved `for_each` and check blocks.
#   - All resources inherit default_tags; never set Project/CostCenter manually.
###############################################################################

terraform {
  required_version = ">= 1.7"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.90"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }


}

provider "aws" {
  region = var.aws_region

  # Apply mandatory tags to every resource automatically.
  # Individual resources must NOT override these; they may add extra tags.
  default_tags {
    tags = {
      Project     = "Chitrakatha"
      CostCenter  = "MLOps-Research"
      Environment = var.environment
      ManagedBy   = "Terraform"
    }
  }
}

###############################################################################
# Data Sources — resolved at plan time, available to all modules
###############################################################################

# Current AWS account ID — used to construct resource names (no hardcoding).
data "aws_caller_identity" "current" {}

# Current region — used to construct ARNs where `var.aws_region` is ambiguous.
data "aws_region" "current" {}
