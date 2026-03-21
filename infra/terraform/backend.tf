###############################################################################
# Project Chitrakatha — Terraform Remote Backend
#
# Why: Moving from local state to S3 allows GitHub Actions to fetch outputs
#      directly, eliminates the need for manual secrets, and provides state
#      locking to prevent concurrent modifications.
###############################################################################

terraform {
  # This block will move your state to S3.
  # NOTE: The bucket-name must exist before you can run `terraform init`.
  # For your first run, you should create the bucket MANUALLY or remove this
  # block until the first apply finishes.
  
  backend "s3" {
    bucket         = "project-chitrakatha-tf-state-54c4cc5f"
    key            = "dev/terraform.tfstate"
    region         = "ap-southeast-2"
    encrypt        = true
    dynamodb_table = "project-chitrakatha-tf-lock"
  }
}
