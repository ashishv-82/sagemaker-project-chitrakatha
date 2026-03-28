# Project Chitrakatha — Progress Tracker

> Last updated: 2026-03-24
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
| **Phase 7** | Architecture migration: pgvector + Bedrock Qwen3 Next 80B A3B | ✅ Complete |

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
| `src/chitrakatha/exceptions.py` | ✅ Done | Full custom exception hierarchy — `PgVectorError` (renamed from `S3VectorError` in Phase 7) |
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
| `infra/terraform/s3.tf` | ✅ Done | 3 buckets (vectors bucket removed in Phase 7) |
| `infra/terraform/faiss_index.tf` | ✅ Deleted | Removed in Phase 7; replaced by `pgvector.tf` |
| `infra/terraform/networking.tf` | ✅ Done | VPC + public/private subnets + VPC endpoints (Bedrock, Secrets Manager, S3) |
| `infra/terraform/rds.tf` | ✅ Done | RDS PostgreSQL 16 (db.t4g.micro) + Secrets Manager credentials — added in Phase 7 |
| `infra/terraform/pgvector.tf` | ✅ Done | pgvector schema reference + locals — added in Phase 7 |
| `infra/terraform/iam.tf` | ✅ Done | SageMaker + Lambda roles; updated in Phase 7 (Lambda: Bedrock + Secrets Manager + VPC NIC policies) |
| `infra/terraform/secrets.tf` | ✅ Done | Secrets Manager secret with placeholder + `ignore_changes` |
| `infra/terraform/cloudwatch.tf` | ✅ Done | 3 alarms + dashboard; `treat_missing_data=notBreaching` |
| `infra/terraform/lambda.tf` | ✅ Done | Lambda + API Gateway; updated in Phase 7 (VPC config, `DB_SECRET_ARN`/`BEDROCK_QWEN3_MODEL_ID` env vars, 60s timeout) |
| `infra/terraform/studio.tf` | ✅ Done | SageMaker Studio domain with idle shutdown |
| `infra/terraform/github_oidc.tf` | ✅ Done | OIDC role for GitHub Actions |
| `infra/terraform/outputs.tf` | ✅ Done | Outputs updated in Phase 7 (removed FAISS, added `rds_endpoint`, `rds_secret_arn`, `lambda_security_group_id`, `private_subnet_ids`) |

---

## Phase 2 — Data Layer & Ingestion ✅

| File | Status | Notes |
|---|---|---|
| `data/scripts/upload_to_bronze.py` | ✅ Done | UTF-8 validated; .txt/.md/.vtt/.xlsx; MD5 checksum in S3 metadata |
| `src/chitrakatha/ingestion/__init__.py` | ✅ Done | Package marker |
| `src/chitrakatha/ingestion/chunker.py` | ✅ Done | Sliding-window 15% overlap, NFC normalization, Devanagari-safe |
| `src/chitrakatha/ingestion/embedder.py` | ✅ Done | Titan Embed v2, batch 25, 3× retry with exponential backoff; 1024-dim output |
| `src/chitrakatha/ingestion/pgvector_writer.py` | ✅ Done | Idempotent pgvector insert (ON CONFLICT DO NOTHING); schema init; Secrets Manager creds — added in Phase 7 |
| `src/chitrakatha/ingestion/faiss_writer.py` | ✅ Deleted | Removed in Phase 7; replaced by `pgvector_writer.py` |
| `data/scripts/ingest_to_faiss.py` | ✅ Deleted | Removed in Phase 7; pgvector insert now in `embed_and_index.py` |
| `data/scripts/synthesize_training_pairs.py` | ✅ Done | Flow B RAFT: golden + 2 distractors + CoT → Gold JSONL |

---

## Phase 3 — SageMaker MLOps Pipeline ✅

| File | Status | Notes |
|---|---|---|
| `pipeline/steps/__init__.py` | ✅ Done | Package marker |
| `pipeline/steps/preprocessing.py` | ✅ Done | Bronze→Silver: NFC norm, SHA-256 dedup, language detection, dual output |
| `pipeline/steps/embed_and_index.py` | ✅ Done | Flow A: corpus → pgvector RDS; updated in Phase 7 (FAISS → pgvector; pip-installs psycopg2-binary) |
| `pipeline/steps/synthesize_pairs.py` | ✅ Done | Flow B step: RAFT synthesis; logs `raft_pairs_generated` + Bedrock tokens |
| `pipeline/steps/train.py` | ✅ Done | QLoRA 4-bit NF4; RAFT prompt; on-demand ml.g4dn.xlarge; Qwen2.5-3B loaded from `SM_CHANNEL_MODEL` (S3 cache) |
| `pipeline/steps/evaluate.py` | ✅ Done | 3 suites: factual (ROUGE-L+BERTScore+EM), cross-lingual, distractor robustness |
| `pipeline/pipeline.py` | ✅ Done | 8-step DAG; dual-threshold ConditionStep; S3+ProcessingInput pattern; `DB_SECRET_ARN` replaces FAISS env vars; `model` input channel passes S3 model cache to TrainingStep; `TrainQLoRARAFT` depends only on `SynthesizePairs` (Flow A and Flow B fully decoupled) |
| `pipeline/requirements.txt` | ✅ Done | Exact-pinned: trl==0.8.6, transformers==4.40.0, torch==2.1.0, etc. |
| `pipeline/Dockerfile` | 🔲 To Do | Custom ECR training image — pre-bakes all deps; eliminates runtime `pip install` in processing steps |

---

## Phase 4 — Serving & Lambda Bridge ✅

| File | Status | Notes |
|---|---|---|
| `serving/deploy_endpoint.py` | ✅ Done | Real-time ml.g4dn.xlarge + scale-to-zero — benchmarking only (not live serving path) |
| `serving/inference.py` | ✅ Done | SageMaker endpoint entry point for fine-tuned Qwen2.5-3B (benchmarking); rewritten in Phase 7 (pgvector retrieval + Qwen2.5-3B generation) |
| `serving/lambda/handler.py` | ✅ Done | Full RAG in Lambda: Titan Embed v2 → pgvector → Bedrock Qwen3 Next 80B A3B; language-aware; rewritten in Phase 7 |
| `serving/lambda/requirements.txt` | ✅ Done | `pydantic>=2`, `psycopg2-binary>=2.9.0`; updated in Phase 7 |
| `infra/terraform/lambda.tf` | ✅ Done | Lambda + API Gateway; VPC config + new env vars added in Phase 7 |

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
| `.github/workflows/ct.yml` | ✅ Done | SageMaker Pipeline trigger on push to main; `rds_secret_arn` replaces FAISS env vars; ECR image build step deferred |
| `.github/workflows/deploy.yml` | ✅ Done | Endpoint deployment — benchmarking only (not live serving path) |

---

## Phase 7 — Architecture Migration: pgvector + Bedrock Qwen3 Next 80B A3B ✅

### Cleanup — Remove Obsolete Infra & Code

| Task | Status | Notes |
|---|---|---|
| Empty S3 Vectors bucket | ✅ Done | Deleted all versions + delete markers manually |
| Remove S3 Vectors bucket from `s3.tf` | ✅ Done | Removed `prevent_destroy = true` and bucket resource |
| Delete `infra/terraform/faiss_index.tf` | ✅ Done | File deleted |
| Remove FAISS outputs from `outputs.tf` | ✅ Done | `s3_vectors_bucket`, `s3_vectors_bucket_arn`, `s3_faiss_index_prefix` removed |
| Remove vectors bucket from `iam.tf` SageMaker policy | ✅ Done | |
| Remove `S3ReadJumpStartPrivateCache` from `iam.tf` | ✅ Done | |
| Remove `sagemaker:InvokeEndpoint` from Lambda policy in `iam.tf` | ✅ Done | |
| Remove `SAGEMAKER_ENDPOINT_NAME` from `lambda.tf` | ✅ Done | Replaced by `DB_SECRET_ARN` + `BEDROCK_QWEN3_MODEL_ID` + `BEDROCK_EMBED_MODEL_ID` |
| Delete `src/chitrakatha/ingestion/faiss_writer.py` | ✅ Done | |
| Delete `data/scripts/ingest_to_faiss.py` | ✅ Done | |
| Delete `tests/unit/test_faiss_writer.py` | ✅ Done | |
| Delete SageMaker real-time endpoint (if deployed) | ✅ Done | Was never deployed; benchmarking endpoint spun up on-demand only |

### New Infrastructure (Terraform — applied 2026-03-24)

| Task | Status | Notes |
|---|---|---|
| `infra/terraform/variables.tf` — add `db_instance_class`, `db_name` | ✅ Done | |
| `infra/terraform/main.tf` — add `random ~> 3.6` provider | ✅ Done | |
| `infra/terraform/networking.tf` — private subnets + VPC endpoints | ✅ Done | 10.0.3/4.0/24; Bedrock runtime + Secrets Manager interface endpoints; S3 gateway endpoint |
| `infra/terraform/rds.tf` — RDS PostgreSQL + pgvector | ✅ Done | `chitrakatha-pgvector.cbqu20w4kpio.ap-southeast-2.rds.amazonaws.com` |
| `infra/terraform/pgvector.tf` — schema reference + locals | ✅ Done | Schema init happens in `pgvector_writer.py` (RDS is private; Terraform can't reach it) |
| `infra/terraform/iam.tf` — Lambda policy update | ✅ Done | Bedrock (Qwen3 + Titan Embed), Secrets Manager, EC2 VPC NIC |
| `infra/terraform/lambda.tf` — VPC config + new env vars | ✅ Done | Private subnets, Lambda SG, 60s timeout |
| `infra/terraform/outputs.tf` — add RDS outputs | ✅ Done | `rds_endpoint`, `rds_secret_arn`, `lambda_security_group_id`, `private_subnet_ids` |

### New Code

| Task | Status | Notes |
|---|---|---|
| `src/chitrakatha/ingestion/pgvector_writer.py` | ✅ Done | Idempotent `ON CONFLICT (source_document, chunk_index) DO NOTHING`; 11 unit tests |
| `tests/unit/test_pgvector_writer.py` | ✅ Done | 11 tests: credentials, connect, schema init, insert, conflict skip, rollback |

### Updated Code

| Task | Status | Notes |
|---|---|---|
| `src/chitrakatha/exceptions.py` | ✅ Done | `S3VectorError` → `PgVectorError` |
| `src/chitrakatha/config.py` | ✅ Done | Removed `s3_vectors_bucket`, `s3_faiss_index_prefix`; added `db_secret_arn` |
| `pipeline/steps/embed_and_index.py` | ✅ Done | FAISS → pgvector; pip-installs `psycopg2-binary` at runtime |
| `pipeline/steps/train.py` | ✅ Done | Loads base model from `SM_CHANNEL_MODEL` (S3 cache); falls back to HF Hub for local runs |
| `pipeline/pipeline.py` | ✅ Done | `DB_SECRET_ARN` replaces FAISS env vars; `model` input channel passes S3 cache to TrainingStep |
| `serving/inference.py` | ✅ Done | Benchmarking SageMaker endpoint: pgvector retrieval + Qwen2.5-3B generation |
| `serving/lambda/handler.py` | ✅ Done | Full RAG: Titan Embed v2 → pgvector → Bedrock Qwen3; language detection; 13 unit tests |
| `serving/lambda/requirements.txt` | ✅ Done | Added `psycopg2-binary>=2.9.0` |
| `pyproject.toml` | ✅ Done | `faiss-cpu` → `psycopg2-binary` + `pgvector` |
| `.github/workflows/ct.yml` | ✅ Done | `rds_secret_arn` replaces FAISS env vars; ECR image build step deferred |

### Fine-tuning Infra

| Task | Status | Notes |
|---|---|---|
| `pipeline/Dockerfile` | 🔲 Deferred | Custom ECR training image — nice-to-have; current approach (runtime pip install) works |
| One-time S3 model cache | ✅ Done | 6.18GB at `s3://chitrakatha-gold-152141418178/base-models/qwen2.5-3b-instruct/` |
| Enable Bedrock Qwen3 Next 80B A3B in AWS console | ✅ Done | Confirmed working via `tests/integration/test_bedrock_models.py` |

---

## Unit Tests ✅

| File | Status | Notes |
|---|---|---|
| `tests/unit/test_chunker.py` | ✅ Done | 10 tests: basic, Devanagari, overlap, typed output, error cases |
| `tests/unit/test_embedder.py` | ✅ Done | 9 tests: batching, dim check, retry, Devanagari passthrough |
| `tests/unit/test_faiss_writer.py` | ✅ Deleted | Removed in Phase 7; replaced by `test_pgvector_writer.py` |
| `tests/unit/test_preprocessor.py` | ✅ Done | 14 tests: language detection, VTT parsing, dedup, process() full flow, error propagation |
| `tests/unit/test_lambda_handler.py` | ✅ Done | 13 tests: valid EN/HI queries, mixed script detection, direct invocation, no-chunks fallback, source dedup, 400/500 error paths, content-type header — rewritten in Phase 7 |

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
