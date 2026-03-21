###############################################################################
# Project Chitrakatha — S3 Vectors Placeholder
#
# Why: Instead of a persistent (expensive) OpenSearch cluster, we implement
#      a production "Scale-to-Zero" RAG using FAISS-over-S3. 
#      The vector index is stored as a binary file in the `vectors` bucket.
#
# Constraints:
#   - Dimension MUST match Titan Embed v2 (1536).
#   - This file provides the metadata and placeholders for the Python code
#     to identify the correct S3 prefix for the index.
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
