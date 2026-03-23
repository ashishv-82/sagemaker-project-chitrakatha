# Project Chitrakatha — Progress Tracker

> Last updated: 2026-03-23
> Reference: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md) · [ARCHITECTURAL_DECISIONS.md](./ARCHITECTURAL_DECISIONS.md)

---

## Phase Summary

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Repo scaffold & governance | ✅ Complete |
| **Phase 1** | Terraform IaC (base) | ✅ Complete |
| **Phase 2** | Data layer (ingestion, chunking, embedding) | ✅ Complete |
| **Phase 3** | SageMaker MLOps pipeline | ✅ Complete |
| **Phase 4** | Serving (original — Qwen endpoint + FAISS) | ✅ Complete |
| **Phase 5** | Observability & lineage | ✅ Complete |
| **Phase 6** | CI/CD (GitHub Actions) | ✅ Complete |
| **Phase 7** | Architecture migration: pgvector + Bedrock Haiku | 🔄 In Progress |

---

## Phase 0 — Repo Scaffold & Governance ✅

| File | Status | Notes |
|---|---|---|
| `pyproject.toml` | ✅ Done | Python 3.12, all deps declared |
| `.python-version` | ✅ Done | Pinned to 3.12 |
| `.pre-commit-config.yaml` | ✅ Done | ruff, mypy, detect-secrets, terraform fmt |
| `Makefile` | ✅ Done | install, lint, test, tf-plan, tf-apply, pipeline-run |
| `src/chitrakatha/__init__.py` | ✅ Done | Exposes `__version__` |
| `src/chitrakatha/config.py` | ✅ Done | Pydantic v2 BaseSettings, no hardcoded values |
| `src/chitrakatha/exceptions.py` | ✅ Done | Full custom exception hierarchy — **`S3VectorError` to be renamed `PgVectorError` in Phase 7** |
| `AGENTS.md` | ✅ Done | Staff MLOps persona & project rules |
| `README.md` | ✅ Done | Architecture diagram, cost breakdown, repo layout |
| `docs/IMPLEMENTATION_PLAN.md` | ✅ Done | Phase-by-phase build plan |
| `tests/__init__.py` | ✅ Done | |
| `tests/unit/__init__.py` | ✅ Done | |
| `tests/integration/__init__.py` | ✅ Done | |
| `.gitignore` | ✅ Done | |
| **GitHub remote** | ✅ Done | `ashishv-82/sagemaker-project-chitrakatha` (private) |

---

## Phase 1 — Infrastructure-as-Code (Terraform) ✅

| File | Status | Notes |
|---|---|---|
| `infra/terraform/main.tf` | ✅ Done | Provider `aws ~> 5.90`, S3 remote backend |
| `infra/terraform/variables.tf` | ✅ Done | 12 typed+validated vars; no secrets or ARNs |
| `infra/terraform/kms.tf` | ✅ Done | CMK, annual rotation, service principal policy |
| `infra/terraform/s3.tf` | ✅ Done | 4 buckets (vectors bucket to be removed in Phase 7) |
| `infra/terraform/faiss_index.tf` | ✅ Done | FAISS S3 prefix local — **to be deleted in Phase 7** |
| `infra/terraform/networking.tf` | ✅ Done | VPC + public subnets — **to be extended in Phase 7** (private subnets + VPC endpoints) |
| `infra/terraform/iam.tf` | ✅ Done | SageMaker + Lambda roles; 7 least-privilege inline policies — **to be updated in Phase 7** |
| `infra/terraform/secrets.tf` | ✅ Done | Secrets Manager secret with placeholder + `ignore_changes` |
| `infra/terraform/cloudwatch.tf` | ✅ Done | 3 alarms + dashboard; `treat_missing_data=notBreaching` |
| `infra/terraform/lambda.tf` | ✅ Done | Lambda function + API Gateway HTTP API trigger — **to be updated in Phase 7** (VPC config, new env vars) |
| `infra/terraform/studio.tf` | ✅ Done | SageMaker Studio domain with idle shutdown |
| `infra/terraform/github_oidc.tf` | ✅ Done | OIDC role for GitHub Actions |
| `infra/terraform/outputs.tf` | ✅ Done | 16 outputs — **to be updated in Phase 7** (remove FAISS, add RDS) |

---

## Phase 2 — Data Layer & Ingestion ✅

| File | Status | Notes |
|---|---|---|
| `data/scripts/upload_to_bronze.py` | ✅ Done | UTF-8 validated; .txt/.md/.vtt/.xlsx; MD5 checksum in S3 metadata |
| `src/chitrakatha/ingestion/__init__.py` | ✅ Done | Package marker |
| `src/chitrakatha/ingestion/chunker.py` | ✅ Done | Sliding-window 15% overlap, NFC normalization, Devanagari-safe |
| `src/chitrakatha/ingestion/embedder.py` | ✅ Done | Titan Embed v2, batch 25, 3× retry with exponential backoff |
| `src/chitrakatha/ingestion/faiss_writer.py` | ✅ Done | FAISS-on-S3 indexer — **to be deleted in Phase 7; replaced by `pgvector_writer.py`** |
| `data/scripts/ingest_to_faiss.py` | ✅ Done | Flow A orchestration: Silver /corpus/ → FAISS-on-S3 — **to be deleted in Phase 7** |
| `data/scripts/synthesize_training_pairs.py` | ✅ Done | Flow B RAFT: golden + 2 distractors + CoT → Gold JSONL |

---

## Phase 3 — SageMaker MLOps Pipeline ✅

| File | Status | Notes |
|---|---|---|
| `pipeline/steps/__init__.py` | ✅ Done | Package marker |
| `pipeline/steps/preprocessing.py` | ✅ Done | Bronze→Silver: NFC norm, SHA-256 dedup, language detection, dual output |
| `pipeline/steps/embed_and_index.py` | ✅ Done | Flow A: corpus→FAISS-on-S3 — **to be updated in Phase 7 (FAISS → pgvector)** |
| `pipeline/steps/synthesize_pairs.py` | ✅ Done | Flow B step: RAFT synthesis; logs `raft_pairs_generated` + Bedrock tokens |
| `pipeline/steps/train.py` | ✅ Done | QLoRA 4-bit NF4; RAFT prompt; on-demand ml.g4dn.xlarge; Qwen2.5-3B; pinned TRL 0.8.6 |
| `pipeline/steps/evaluate.py` | ✅ Done | 3 suites: factual (ROUGE-L+BERTScore+EM), cross-lingual, distractor robustness |
| `pipeline/pipeline.py` | ✅ Done | 8-step DAG; dual-threshold ConditionStep; S3+ProcessingInput pattern; no hardcoded ARNs — **to be updated in Phase 7 (HuggingFace estimator + SM_CHANNEL_MODEL)** |
| `pipeline/requirements.txt` | ✅ Done | Exact-pinned: trl==0.8.6, transformers==4.40.0, torch==2.1.0, etc. |

---

## Phase 4 — Serving & Lambda Bridge ✅

| File | Status | Notes |
|---|---|---|
| `serving/deploy_endpoint.py` | ✅ Done | Real-time ml.g4dn.xlarge + scale-to-zero — **repurposed in Phase 7: benchmarking only, not live serving** |
| `serving/inference.py` | ✅ Done | RAG: embed → FAISS-on-S3 → Qwen — **to be rewritten in Phase 7 (pgvector + Bedrock Haiku)** |
| `serving/lambda/handler.py` | ✅ Done | Language-aware, pydantic validation, 503 cold-start handling — **to be updated in Phase 7 (new contract, VPC, Bedrock)** |
| `serving/lambda/requirements.txt` | ✅ Done | `boto3`, `pydantic>=2` — **to be updated in Phase 7 (add psycopg2-binary, pgvector)** |
| `infra/terraform/lambda.tf` | ✅ Done | Lambda + API Gateway HTTP API — **to be updated in Phase 7 (VPC config, new env vars)** |

---

## Phase 5 — Observability & Lineage ✅

| File | Status | Notes |
|---|---|---|
| `src/chitrakatha/monitoring/lineage.py` | ✅ Done | SageMaker Lineage chain |
| `src/chitrakatha/monitoring/experiments.py` | ✅ Done | Hyperparams + eval metrics logger |
| `infra/terraform/cloudwatch.tf` (additions) | ✅ Done | Dashboard: invocations, error rate, cold-start |

---

## Phase 6 — CI/CD (GitHub Actions) ✅

| File | Status | Notes |
|---|---|---|
| `.github/workflows/ci.yml` | ✅ Done | Lint, type-check, unit tests |
| `.github/workflows/tf-check.yml` | ✅ Done | Terraform formatting, validate, and tfsec |
| `.github/workflows/ct.yml` | ✅ Done | SageMaker Pipeline trigger on push to main — **to be updated in Phase 7 (ECR image build step)** |
| `.github/workflows/deploy.yml` | ✅ Done | Endpoint deployment — **repurposed in Phase 7: benchmarking only** |

---

## Phase 7 — Architecture Migration: pgvector + Bedrock Qwen3 Next 80B A3B 🔄

### Cleanup — Remove Obsolete Infra & Code

| Task | Status | Notes |
|---|---|---|
| Empty S3 Vectors bucket | 🔲 To Do | Must empty before Terraform can destroy |
| Remove S3 Vectors bucket from `s3.tf` | 🔲 To Do | Remove `prevent_destroy = true` first |
| Delete `infra/terraform/faiss_index.tf` | 🔲 To Do | Entire file — only contains `s3_faiss_index_prefix` local |
| Remove FAISS outputs from `outputs.tf` | 🔲 To Do | Remove `s3_vectors_bucket`, `s3_vectors_bucket_arn`, `s3_faiss_index_prefix` |
| Remove vectors bucket from `iam.tf` SageMaker policy | 🔲 To Do | Remove from `S3ReadWriteProjectBuckets` statement |
| Remove `S3ReadJumpStartPrivateCache` from `iam.tf` | 🔲 To Do | No longer using JumpStart |
| Remove `sagemaker:InvokeEndpoint` from Lambda policy in `iam.tf` | 🔲 To Do | Lambda calls Bedrock directly now |
| Remove `SAGEMAKER_ENDPOINT_NAME` from `lambda.tf` | 🔲 To Do | Replaced by `DB_SECRET_ARN` + `BEDROCK_QWEN3_MODEL_ID` |
| Delete `src/chitrakatha/ingestion/faiss_writer.py` | 🔲 To Do | Replaced by `pgvector_writer.py` |
| Delete `data/scripts/ingest_to_faiss.py` | 🔲 To Do | Flow A FAISS orchestration; replaced by pgvector insert in `embed_and_index.py` |
| Delete `tests/unit/test_faiss_writer.py` | 🔲 To Do | Replaced by `tests/unit/test_pgvector_writer.py` |
| Delete SageMaker real-time endpoint (if deployed) | 🔲 To Do | Not needed for live traffic; spin up on-demand for benchmarking |

### New Infrastructure (Terraform)

| Task | Status | Notes |
|---|---|---|
| `infra/terraform/variables.tf` — add `db_instance_class`, `db_name` | 🔲 To Do | |
| `infra/terraform/main.tf` — add `random ~> 3.6` provider | 🔲 To Do | For RDS password generation |
| `infra/terraform/networking.tf` — private subnets + VPC endpoints | 🔲 To Do | Private subnets 10.0.3.0/24, 10.0.4.0/24; VPC endpoints: Bedrock runtime + Secrets Manager (interface) + S3 (gateway, free) |
| `infra/terraform/rds.tf` — RDS PostgreSQL + pgvector | 🔲 To Do | db.t4g.micro, PostgreSQL 16, KMS-encrypted, private subnet, credentials in Secrets Manager |
| `infra/terraform/pgvector.tf` — schema init | 🔲 To Do | Creates `vector` extension, `embeddings` table, HNSW index |
| `infra/terraform/iam.tf` — Lambda policy update | 🔲 To Do | Add: Bedrock (Qwen3 Next 80B A3B + Titan Embed), Secrets Manager (rds_credentials), EC2 VPC network interface |
| `infra/terraform/lambda.tf` — VPC config + new env vars | 🔲 To Do | Add `vpc_config`, replace env vars, increase timeout to 60s |
| `infra/terraform/outputs.tf` — add RDS outputs | 🔲 To Do | Add `rds_endpoint`, `rds_secret_arn`, `lambda_security_group_id`, `private_subnet_ids` |

### New Code

| Task | Status | Notes |
|---|---|---|
| `src/chitrakatha/ingestion/pgvector_writer.py` | 🔲 To Do | psycopg2 + pgvector; idempotent upsert; raises `PgVectorError` |
| `tests/unit/test_pgvector_writer.py` | 🔲 To Do | Replaces `test_faiss_writer.py` |

### Updated Code

| Task | Status | Notes |
|---|---|---|
| `src/chitrakatha/exceptions.py` | 🔲 To Do | Rename `S3VectorError` → `PgVectorError` |
| `pipeline/steps/embed_and_index.py` | 🔲 To Do | Replace FAISS S3 upload with pgvector insert |
| `pipeline/pipeline.py` | 🔲 To Do | Switch training step from JumpStart to HuggingFace estimator with custom ECR image; add `SM_CHANNEL_MODEL` input from S3 Gold base model cache |
| `serving/inference.py` | 🔲 To Do | Rewrite: pgvector retrieval + Bedrock Qwen3 Next 80B A3B generation; RDS connection pooled at module level |
| `serving/lambda/handler.py` | 🔲 To Do | New request/response contract (`query`, `session_id`, `history`); remove 503 cold-start logic; Lambda in VPC |
| `serving/lambda/requirements.txt` | 🔲 To Do | Add `psycopg2-binary`, `pgvector` |
| `.github/workflows/ct.yml` | 🔲 To Do | Add ECR Docker image build + push step (triggered when `pipeline/requirements.txt` changes) |

### Fine-tuning Infra

| Task | Status | Notes |
|---|---|---|
| `pipeline/Dockerfile` | 🔲 To Do | Custom ECR training image; pre-bakes all `pipeline/requirements.txt` deps |
| One-time S3 model cache | 🔲 To Do | `huggingface-cli download Qwen/Qwen2.5-3B-Instruct` → `aws s3 sync s3://chitrakatha-gold/base-models/qwen2.5-3b-instruct/` |
| Enable Bedrock Qwen3 Next 80B A3B in AWS console | 🔲 To Do | One-time manual step in ap-southeast-2 |

---

## Unit Tests ✅

| File | Status | Notes |
|---|---|---|
| `tests/unit/test_chunker.py` | ✅ Done | 10 tests: basic, Devanagari, overlap, typed output, error cases |
| `tests/unit/test_embedder.py` | ✅ Done | 9 tests: batching, dim check, retry, Devanagari passthrough |
| `tests/unit/test_faiss_writer.py` | ✅ Done | 8 tests — **to be deleted in Phase 7; replaced by `test_pgvector_writer.py`** |
| `tests/unit/test_preprocessor.py` | ✅ Done | 14 tests: language detection, VTT parsing, dedup, process() full flow, error propagation |
| `tests/unit/test_lambda_handler.py` | ✅ Done | 12 tests: valid EN/HI queries, language detection, 400/500 error paths — **to be updated in Phase 7 (new contract)** |

---

## Fixes & Improvements (post-Phase 6)

| Change | File(s) | Notes |
|---|---|---|
| Add `BedrockSynthesisError` | `src/chitrakatha/exceptions.py`, `data/scripts/synthesize_training_pairs.py`, `pipeline/steps/synthesize_pairs.py` | Separate exception for Claude synthesis failures vs. Titan embedding failures |
| FAISS index size guard | `src/chitrakatha/ingestion/faiss_writer.py` | Logs warning when index exceeds 50k vectors |
| Switch to SageMaker JumpStart (Llama 3.2 3B) | `pipeline/pipeline.py`, `pipeline/steps/train.py` | Replaced `HuggingFace` estimator with `JumpStartEstimator` — later reversed in favour of Qwen2.5-3B from HuggingFace Hub |
| Switch to Qwen2.5-3B-Instruct | `pipeline/pipeline.py`, `pipeline/steps/train.py` | Replaced Llama/JumpStart with Qwen2.5-3B (Apache 2.0, better Hindi); downloaded from HuggingFace Hub at training start |
| Pin training framework versions | `pipeline/requirements.txt`, `pipeline/steps/train.py` | TRL 0.29 / transformers 5.x had breaking API changes; pinned to trl==0.8.6, transformers==4.40.0, torch==2.1.0 |
| Switch to on-demand training | `pipeline/pipeline.py` | `use_spot_instances=False`; Spot quota is 0 in ap-southeast-2 |
| Processing container package fix | `pipeline/pipeline.py` | `_sync_chitrakatha_to_s3()` uploads `src/chitrakatha/` to Silver bucket; each step mounts via `chitrakatha_input` ProcessingInput; `PYTHONPATH=/opt/ml/processing/input/src` |
| Training container package fix | `pipeline/pipeline.py` | `_make_training_source_dir()` copies full `pipeline/steps/` + `chitrakatha/` package into estimator `source_dir` |
| Switch to real-time endpoint + scale-to-zero | `serving/deploy_endpoint.py`, `serving/lambda/handler.py` | Replaced `ServerlessInferenceConfig` with real-time `ml.g4dn.xlarge`; App Auto Scaling `MinCapacity=0`; Lambda returns HTTP 503 + `Retry-After: 300` |
| Instance types aligned to 3B model | `pipeline/pipeline.py` | Training + eval both use `ml.g4dn.xlarge` (16 GB VRAM, T4); significant cost reduction per run |
| LoRA adapter merge | `pipeline/steps/train.py` | `trainer.model.merge_and_unload()` before `save_pretrained()` — without this endpoint serves unmodified base model |
| Remove `source_dir` — S3+ProcessingInput pattern | `pipeline/pipeline.py` | `source_dir` not reliably propagated through `@runnable_by_pipeline` in SDK ≥2.200; fixed with S3 sync + ProcessingInput pattern |
| KMS GenerateDataKey for GitHub Actions | `infra/terraform/github_oidc.tf` | `_sync_chitrakatha_to_s3()` writes to KMS-encrypted Silver bucket; added `KMSEncryptDecryptForS3Uploads` statement |
| sagemaker:AddTags + CreatePipeline for GitHub Actions | `infra/terraform/github_oidc.tf` | `pipeline.upsert()` calls `CreatePipeline` and tags it; added `CreatePipeline`, `UpdatePipeline`, `AddTags` |
| sagemaker:AddTags + pipelines-* scope for execution role | `infra/terraform/iam.tf` | Pipeline auto-tags spawned jobs with `pipelines-<execid>-*` pattern; added `AddTags` and second resource ARN pattern |
| SageMaker Studio domain | `infra/terraform/studio.tf` | Added Studio domain with IAM auth, default user profile, 60-min idle timeout |
| Replace piecemeal Studio policies with AmazonSageMakerFullAccess | `infra/terraform/iam.tf` | Eliminated drip-feed of individual Studio permissions; attached AWS managed policy |
| iam:PassRole self-pass | `infra/terraform/iam.tf` | Pipeline orchestrator needs to pass execution role to child ProcessingJob/TrainingJob |
| Install packages in processing containers | `pipeline/steps/preprocessing.py`, `pipeline/steps/embed_and_index.py`, `pipeline/steps/synthesize_pairs.py` | SKLearnProcessor/ScriptProcessor base containers don't pre-install pydantic, faiss-cpu, numpy; added pip install block at top of each step |
