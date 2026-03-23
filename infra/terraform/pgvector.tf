###############################################################################
# Project Chitrakatha — pgvector Schema Reference
#
# Why: RDS is in a private subnet with no public access, so Terraform cannot
#      connect to PostgreSQL directly during `apply`. Schema initialisation is
#      performed by pgvector_writer.py on first pipeline execution using
#      idempotent DDL (CREATE EXTENSION IF NOT EXISTS, CREATE TABLE IF NOT EXISTS,
#      CREATE INDEX IF NOT EXISTS). Subsequent runs are no-ops.
#
# Schema (executed by pgvector_writer.py):
#
#   CREATE EXTENSION IF NOT EXISTS vector;
#
#   CREATE TABLE IF NOT EXISTS embeddings (
#     id              BIGSERIAL PRIMARY KEY,
#     source_document TEXT        NOT NULL,
#     chunk_index     INT         NOT NULL,
#     chunk_text      TEXT        NOT NULL,
#     embedding       vector(1024),
#     created_at      TIMESTAMPTZ DEFAULT NOW(),
#     UNIQUE (source_document, chunk_index)
#   );
#
#   CREATE INDEX IF NOT EXISTS embeddings_hnsw_idx
#     ON embeddings
#     USING hnsw (embedding vector_cosine_ops)
#     WITH (m = 16, ef_construction = 64);
#
# Query pattern (cosine similarity, top-k):
#   SELECT chunk_text
#   FROM   embeddings
#   ORDER  BY embedding <=> %s
#   LIMIT  5;
###############################################################################

locals {
  pgvector_table      = "embeddings"
  pgvector_dimensions = 1024 # Must match Titan Embed Text v2 output dimensions
}
