###############################################################################
# Project Chitrakatha — S3 Vectors Index
#
# Why: The S3 Vectors index is the serverless RAG knowledge base. Instead of
#      running a persistent OpenSearch or Pinecone cluster (violating the
#      Scale-to-Zero constraint), we store 1536-dim Titan Embed v2 vectors
#      natively in S3 and query them via the boto3 S3 Vectors API.
#
# Provider requirement:
#   hashicorp/aws >= 5.90 is required for the aws_s3_vectors_index resource.
#   The `terraform {}` block in main.tf pins to ~> 5.90.
#
# Constraints:
#   - Dimension MUST match the embedding model output (1536 for Titan Embed v2).
#   - Metric must match what the embedder uses at query time (cosine similarity).
#   - The vectors bucket (aws_s3_bucket.vectors) must exist before this resource.
###############################################################################

locals {
  s3_vector_index_name = "${var.project_name}-rag-index"
  s3_vector_index_arn  = "${aws_s3_bucket.vectors.arn}/index/${local.s3_vector_index_name}"
}

# The aws_s3_vectors_index resource is a simulated 2026 feature. 
# It is not supported in the AWS provider yet, so we use a null_resource placeholder
# to allow Terraform to plan and apply the rest of the infrastructure.
resource "null_resource" "chitrakatha_rag_index" {
  triggers = {
    bucket    = aws_s3_bucket.vectors.id
    name      = local.s3_vector_index_name
    dimension = var.s3_vector_dimension
    metric    = var.s3_vector_metric
  }
}
