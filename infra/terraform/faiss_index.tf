###############################################################################
# Project Chitrakatha — FAISS on S3 Placeholder
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
  s3_faiss_index_prefix = "index/${var.project_name}-rag-index"
}

