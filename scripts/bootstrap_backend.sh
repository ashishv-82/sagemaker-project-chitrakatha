#!/usr/bin/env bash

# Project Chitrakatha — Terraform Backend Bootstrap
#
# This script provisions the S3 Bucket and DynamoDB table required by Terraform
# to store and lock state remotely. Run this script ONCE before running `terraform init`.

set -e

REGION="${1:-ap-southeast-2}"
UUID=$(uuidgen | tr '[:upper:]' '[:lower:]' | cut -c 1-8)
STATE_BUCKET="chitrakatha-tf-state-${UUID}"
LOCK_TABLE="chitrakatha-tf-lock"

echo "============================================================"
echo "🚀 Bootstrapping Terraform Backend for Project Chitrakatha"
echo "Region:       $REGION"
echo "State Bucket: $STATE_BUCKET"
echo "Lock Table:   $LOCK_TABLE"
echo "============================================================"

# 1. Create S3 Bucket
echo "Creating S3 Bucket for Terraform state..."
if [ "$REGION" == "us-east-1" ]; then
    aws s3api create-bucket \
        --bucket "$STATE_BUCKET" \
        --region "$REGION"
else
    aws s3api create-bucket \
        --bucket "$STATE_BUCKET" \
        --region "$REGION" \
        --create-bucket-configuration LocationConstraint="$REGION"
fi

# 2. Enable Bucket Versioning
echo "Enabling S3 Bucket versioning (essential for state recovery)..."
aws s3api put-bucket-versioning \
    --bucket "$STATE_BUCKET" \
    --versioning-configuration Status=Enabled

# 3. Create DynamoDB Lock Table
echo "Creating DynamoDB table for state locking..."
aws dynamodb create-table \
    --table-name "$LOCK_TABLE" \
    --attribute-definitions AttributeName=LockID,AttributeType=S \
    --key-schema AttributeName=LockID,KeyType=HASH \
    --billing-mode PAY_PER_REQUEST \
    --region "$REGION" > /dev/null

echo "Waiting for DynamoDB table to become ACTIVE..."
aws dynamodb wait table-exists --table-name "$LOCK_TABLE" --region "$REGION"

echo ""
echo "✅ Bootstrap Complete!"
echo "You can now initialize Terraform using the following command:"
echo ""
echo "cd infra/terraform"
echo "terraform init \\"
echo "  -backend-config=\"bucket=${STATE_BUCKET}\" \\"
echo "  -backend-config=\"region=${REGION}\" \\"
echo "  -backend-config=\"dynamodb_table=${LOCK_TABLE}\""
echo ""
