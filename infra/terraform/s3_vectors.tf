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

resource "aws_s3_vectors_index" "chitrakatha_rag" {
  # Which S3 Vectors bucket hosts this index.
  bucket = aws_s3_bucket.vectors.id

  # Index name — referenced in outputs and injected into the Lambda/endpoint env.
  name = "${var.project_name}-rag-index"

  # Must match Titan Embed Text v2 output dimensionality.
  dimension = var.s3_vector_dimension

  # Cosine similarity is best for semantic search on normalised text embeddings.
  distance_metric = var.s3_vector_metric

  # Encrypt the index with the same CMK as the rest of the data lake.
  encryption_configuration {
    kms_key_arn = aws_kms_key.chitrakatha.arn
  }
}
