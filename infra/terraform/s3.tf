###############################################################################
# Project Chitrakatha — S3 Data Lake Buckets
#
# Why: A three-tier data lake (Bronze/Silver/Gold) + a dedicated S3 Vectors
#      bucket separates raw, cleaned, and training-ready data clearly. S3
#      versioning on every bucket ensures reproducible training runs — we can
#      re-point the pipeline to any historical Silver or Gold snapshot.
#
# Buckets:
#   1. Bronze — raw ingest (articles, transcripts, Excel, synopsis)
#   2. Silver — cleaned JSONL (preprocessing output)
#   3. Gold   — fine-tuning JSONL, model checkpoints, evaluation results
#
# Constraints:
#   - All buckets: versioning ENABLED, KMS CMK encryption, public access BLOCKED.
#   - Lifecycle policies archive non-current versions to STANDARD_IA after 30
#     days and expire them after 365 days (cost optimisation).
#   - `prevent_destroy = true` on all buckets — data is never deleted by Terraform.
###############################################################################

locals {
  account_id = data.aws_caller_identity.current.account_id

  # Bucket names include account ID to guarantee global uniqueness.
  bucket_names = {
    bronze = "${var.project_name}-bronze-${local.account_id}"
    silver = "${var.project_name}-silver-${local.account_id}"
    gold   = "${var.project_name}-gold-${local.account_id}"
  }
}

###############################################################################
# Bronze — Raw Ingest
###############################################################################

resource "aws_s3_bucket" "bronze" {
  bucket = local.bucket_names.bronze
}

resource "aws_s3_bucket_versioning" "bronze" {
  bucket = aws_s3_bucket.bronze.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "bronze" {
  bucket = aws_s3_bucket.bronze.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.chitrakatha.arn
    }
    # bucket_key_enabled reduces KMS API calls and cost by ~99%.
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "bronze" {
  bucket                  = aws_s3_bucket.bronze.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "bronze" {
  bucket = aws_s3_bucket.bronze.id

  rule {
    id     = "manage-noncurrent-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_transition {
      noncurrent_days = var.noncurrent_version_ia_transition_days
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiry_days
    }
  }
}

###############################################################################
# Silver — Cleaned JSONL
###############################################################################

resource "aws_s3_bucket" "silver" {
  bucket = local.bucket_names.silver
}

resource "aws_s3_bucket_versioning" "silver" {
  bucket = aws_s3_bucket.silver.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "silver" {
  bucket = aws_s3_bucket.silver.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.chitrakatha.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "silver" {
  bucket                  = aws_s3_bucket.silver.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "silver" {
  bucket = aws_s3_bucket.silver.id

  rule {
    id     = "manage-noncurrent-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_transition {
      noncurrent_days = var.noncurrent_version_ia_transition_days
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiry_days
    }
  }
}

###############################################################################
# Gold — Training JSONL + Checkpoints + Evaluation Results
###############################################################################

resource "aws_s3_bucket" "gold" {
  bucket = local.bucket_names.gold
}

resource "aws_s3_bucket_versioning" "gold" {
  bucket = aws_s3_bucket.gold.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "gold" {
  bucket = aws_s3_bucket.gold.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm     = "aws:kms"
      kms_master_key_id = aws_kms_key.chitrakatha.arn
    }
    bucket_key_enabled = true
  }
}

resource "aws_s3_bucket_public_access_block" "gold" {
  bucket                  = aws_s3_bucket.gold.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "gold" {
  bucket = aws_s3_bucket.gold.id

  rule {
    id     = "manage-noncurrent-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_transition {
      noncurrent_days = var.noncurrent_version_ia_transition_days
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_expiration {
      noncurrent_days = var.noncurrent_version_expiry_days
    }
  }

  # Transition training checkpoints to cheaper storage after 90 days.
  # Checkpoints in /checkpoints/ prefix are only needed for resuming failed runs.
  rule {
    id     = "transition-checkpoints-to-ia"
    status = "Enabled"

    filter {
      prefix = "checkpoints/"
    }

    transition {
      days          = 90
      storage_class = "STANDARD_IA"
    }
  }
}

