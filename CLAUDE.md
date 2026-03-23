# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Project Chitrakatha** is a bilingual (English + Devanagari Hindi) Indian Comic History LLM on AWS SageMaker. It is 100% serverless (~$5/month baseline) using RAFT (Retrieval-Augmented Fine-Tuning) to fine-tune Llama 3.2 3B Instruct on domain-specific Q&A pairs synthesized by Claude 3.5 Sonnet.

**Two-speed architecture:**
- **Flow A (website):** Titan Embed v2 → pgvector RDS → Bedrock Qwen3 Next 80B A3B → Lambda + API Gateway
- **Flow B (benchmarking):** Claude 3.5 Sonnet → RAFT training pairs → QLoRA fine-tune Qwen2.5-3B → evaluate

**Data flow:** S3 Bronze (raw PDFs/VTTs) → SKLearnProcessor Preprocessing → S3 Silver (JSONL chunks) → parallel:
- **Flow A:** Titan Embed v2 → pgvector RDS (RDS PostgreSQL 16 + pgvector, private subnet, Option B networking)
- **Flow B:** Claude 3.5 Sonnet → RAFT Q&A pairs (90% train / 10% eval split) → QLoRA Qwen2.5-3B

## Commands

```bash
make install      # pip install -e ".[dev]" + pre-commit install
make lint         # ruff check + ruff format --check + mypy src/
make test         # pytest tests/unit/ -v --cov=src/chitrakatha --cov-fail-under=80
make clean        # remove caches and egg-info
make tf-init      # terraform init in infra/terraform/
make tf-plan      # terraform plan
make tf-apply     # terraform apply
make pipeline-run # python pipeline/pipeline.py --execute
```

Run a single test file: `pytest tests/unit/test_pgvector_writer.py -v`

## Architecture

### Core Library (`src/chitrakatha/`)
- **`config.py`** — Pydantic v2 `BaseSettings`. All config is environment-variable-driven (no hardcoded values). `get_settings()` is LRU-cached. All SageMaker Processing Job steps instantiate `Settings()` directly.
- **`exceptions.py`** — Custom hierarchy rooted at `ChitrakathaBaseError`: `BedrockSynthesisError`, `BedrockEmbeddingError`, `PgVectorError`, `DataIngestionError`, `SageMakerPipelineError`.
- **`ingestion/embedder.py`** — Titan Embed Text v2 wrapper. Max batch size 25 (Bedrock API hard limit). Output is 1024-dim vectors.
- **`ingestion/pgvector_writer.py`** — Writes embeddings to pgvector RDS. Schema init (`CREATE EXTENSION/TABLE/INDEX IF NOT EXISTS`) runs on every call (idempotent). Idempotency key: `UNIQUE(source_document, chunk_index)`. Credentials fetched from Secrets Manager via `DB_SECRET_ARN`.
- **`monitoring/experiments.py`** — Logs metrics to SageMaker Experiments.

### Pipeline (`pipeline/pipeline.py`)
SageMaker Pipeline DAG using `PipelineSession`. Steps use `ProcessingStep(step_args=processor.run(...))` — **not** `get_run_args()`. When `PipelineSession` is active, `processor.run()` returns `RunArgs` via the `@runnable_by_pipeline` decorator instead of submitting a job.

**Critical**: `source_dir` must **not** be used in any `processor.run()` call. Neither `ScriptProcessor` nor `FrameworkProcessor` subclasses (`SKLearnProcessor`, `PyTorchProcessor`) reliably propagate `source_dir` through the `@runnable_by_pipeline` replay phase in SageMaker SDK ≥2.200. Instead: `_sync_chitrakatha_to_s3()` in `main()` uploads `src/chitrakatha/` to `s3://{SILVER_BUCKET}/pipeline-assets/src/`, each step mounts it via `chitrakatha_input` (`ProcessingInput`), and `PYTHONPATH=/opt/ml/processing/input/src` is set in `_CONTAINER_ENV`. Step scripts are passed as absolute paths: `code=str(STEPS_DIR / "script.py")`.

`_make_processing_source_dir(script_path)` bundles each step script into a temp dir. It also copies a `<script_name>_requirements.txt` companion file if present (e.g., `preprocessing_requirements.txt` for openpyxl). The temp dir is cleaned up after `pipeline.upsert()`.

Pipeline constructor does **not** accept `tags=`. Tags are applied via `pipeline.upsert(role_arn=..., tags=RESOURCE_TAGS)` only.

Base model: `Qwen/Qwen2.5-3B-Instruct` (HuggingFace Hub, Apache 2.0). Cached to S3 Gold bucket at `base-models/qwen2.5-3b-instruct/` via `scripts/cache_base_model_to_s3.py`. Training uses `PyTorchEstimator` with `SM_CHANNEL_MODEL` to load weights from S3 (no internet download at runtime).

### Serving (`serving/`)
- **`inference.py`** — SageMaker endpoint entry point for the fine-tuned Qwen2.5-3B model (benchmarking path only). Loads model from `model_dir` and generates answers with retrieved context.
- **`deploy_endpoint.py`** — Deploys serverless endpoint + two App Auto Scaling policies.
- **`lambda/handler.py`** — **Primary serving path.** API Gateway bridge that does the full RAG loop: embed query (Bedrock Titan Embed v2) → retrieve top-5 from pgvector RDS → generate with Bedrock Qwen3 Next 80B A3B (`qwen.qwen3-next-80b-a3b`). Detects Devanagari to tag `language` as `"hi"`. Lambda env vars: `DB_SECRET_ARN`, `BEDROCK_QWEN3_MODEL_ID`, `BEDROCK_EMBED_MODEL_ID`. Deployment package in `serving/lambda/package/` + `function.zip`; rebuilt by `terraform apply` via `null_resource.pip_install_lambda`.

### Infrastructure (`infra/terraform/`)
- Terraform ≥1.7, AWS provider ~5.90, region `ap-southeast-2`
- KMS CMKs on all S3 buckets; no hardcoded account IDs (uses `data.aws_caller_identity.current`)
- GitHub Actions authenticates via OIDC (`github_oidc.tf`) — no long-lived IAM keys
- Remote state: S3 bucket + DynamoDB lock table

### CI/CD (`.github/workflows/`)
- **`ci.yml`** — Quality gate on PRs: ruff lint/format, mypy, pytest
- **`ct.yml`** — Continuous Training on push to `main`: fetches Terraform outputs as env vars, runs `python pipeline/pipeline.py --execute`
- **`deploy.yml`** — Manual/event-triggered: runs `python serving/deploy_endpoint.py`
- **`tf-check.yml`** — Terraform fmt/validate/tfsec on infra changes

## Key Constraints

- In commit messages do not add this text "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
- **Python 3.12+**, strict typing (mypy strict mode), pydantic v2 throughout
- **100% serverless** — no EC2, EKS, or persistent databases; scale-to-zero everything
- **UTF-8 everywhere** — Devanagari text is first-class; use `ensure_ascii=False` in `json.dumps`
- **KMS encryption** on all S3 writes; IAM least-privilege
- **Managed Spot Training** with checkpointing in `train.py`
- **`source_document`** field is the canonical chunk grouping key (always written by preprocessing.py). Fields `source_entity` and `publisher` do not exist in the Silver JSONL schema.
- **Distractor selection in synthesize_pairs.py** prefers chunks from different `source_document` values; falls back to any other chunk if insufficient cross-document candidates
- All resource tags: `Project: Chitrakatha`, `CostCenter: MLOps-Research`
- Line length: 100 chars (ruff)
