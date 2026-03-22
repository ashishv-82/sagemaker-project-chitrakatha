# Project Chitrakatha — Progress Tracker

> Last updated: 2026-03-21  
> Reference: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

---

## Phase Summary

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Repo scaffold & governance | ✅ Complete |
| **Phase 1** | Terraform IaC | ✅ Complete |
| **Phase 2** | Data layer (FAISS-on-S3) | ✅ Complete |
| **Phase 3** | SageMaker MLOps pipeline | ✅ Complete |
| **Phase 4** | Serving (FAISS-on-S3) | ✅ Complete |
| **Phase 5** | Observability & lineage | ✅ Complete |
| **Phase 6** | CI/CD (GitHub Actions) | ✅ Complete |

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
| `src/chitrakatha/exceptions.py` | ✅ Done | Full custom exception hierarchy |
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
| `infra/terraform/kms.tf` | ✅ Done | CMK, annual rotation, service principal policy (no circular dep) |
| `infra/terraform/s3.tf` | ✅ Done | 4 buckets: versioning, KMS, public-access-block, lifecycle |
| `infra/terraform/s3_vectors.tf` | ✅ Done | S3 Vectors index (1536-dim, cosine, KMS-encrypted) |
| `infra/terraform/iam.tf` | ✅ Done | SageMaker + Lambda roles; 7 least-privilege inline policies |
| `infra/terraform/secrets.tf` | ✅ Done | Secrets Manager secret with placeholder + `ignore_changes` |
| `infra/terraform/cloudwatch.tf` | ✅ Done | 3 alarms + dashboard; `treat_missing_data=notBreaching` |
| `infra/terraform/outputs.tf` | ✅ Done | 16 outputs: all ARNs, bucket names, dashboard URL |

---

## Phase 2 — Data Layer & Ingestion ✅

| File | Status | Notes |
|---|---|---|
| `data/scripts/upload_to_bronze.py` | ✅ Done | UTF-8 validated; .txt/.md/.vtt/.xlsx; MD5 checksum in S3 metadata |
| `src/chitrakatha/ingestion/__init__.py` | ✅ Done | Package marker |
| `src/chitrakatha/ingestion/chunker.py` | ✅ Done | Sliding-window 15% overlap, NFC normalization, Devanagari-safe |
| `src/chitrakatha/ingestion/embedder.py` | ✅ Done | Titan Embed v2, batch 25, 3× retry with exponential backoff |
| `src/chitrakatha/ingestion/faiss_writer.py` | ✅ Done | Refactored: FAISS-on-S3 indexer (working production RAG) |
| `data/scripts/ingest_to_faiss.py` | ✅ Done | Flow A orchestration: Silver /corpus/ → FAISS-on-S3 |
| `data/scripts/synthesize_training_pairs.py` | ✅ Done | Flow B RAFT: golden + 2 distractors + CoT → Gold JSONL |

---

## Phase 3 — SageMaker MLOps Pipeline ✅

| File | Status | Notes |
|---|---|---|
| `pipeline/steps/__init__.py` | ✅ Done | Package marker |
| `pipeline/steps/preprocessing.py` | ✅ Done | Bronze→Silver: NFC norm, SHA-256 dedup, language detection, dual output |
| `pipeline/steps/embed_and_index.py` | ✅ Done | Flow A: corpus→FAISS-on-S3; idempotent; logs `vector_count_written` |
| `pipeline/steps/synthesize_pairs.py` | ✅ Done | Flow B step: RAFT synthesis; logs `raft_pairs_generated` + Bedrock tokens |
| `pipeline/steps/train.py` | ✅ Done | QLoRA 4-bit NF4; RAFT prompt with document shuffle; Spot training |
| `pipeline/steps/evaluate.py` | ✅ Done | 3 suites: factual (ROUGE-L+BERTScore+EM), cross-lingual, distractor robustness |
| `pipeline/pipeline.py` | ✅ Done | 8-step DAG; dual-threshold ConditionStep; Spot training; no hardcoded ARNs |
| `pipeline/requirements.txt` | ✅ Done | peft, trl, datasets, bitsandbytes, rouge-score, bert-score, sentencepiece |

---

## Phase 4 — Serving & Lambda Bridge ✅

| File | Status | Notes |
|---|---|---|
| `serving/deploy_endpoint.py` | ✅ Done | ServerlessInferenceConfig (6144MB, max 5) |
| `serving/inference.py` | ✅ Done | RAG predict_fn: embed → FAISS-on-S3 → Llama |
| `serving/lambda/handler.py` | ✅ Done | Language-aware, pydantic validation |
| `serving/lambda/requirements.txt` | ✅ Done | `boto3`, `pydantic>=2` |
| `infra/terraform/lambda.tf` | ✅ Done | Lambda function + API Gateway HTTP API trigger |

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
| `.github/workflows/ct.yml` | ✅ Done | SageMaker Pipeline trigger on push to main |
| `.github/workflows/deploy.yml` | ✅ Done | Serverless Endpoint deployment |

---

## Unit Tests ✅

| File | Status | Notes |
|---|---|---|
| `tests/unit/test_chunker.py` | ✅ Done | 10 tests: basic, Devanagari, overlap, typed output, error cases |
| `tests/unit/test_embedder.py` | ✅ Done | 9 tests: batching, dim check, retry, Devanagari passthrough |
| `tests/unit/test_faiss_writer.py` | ✅ Done | 8 tests: idempotency, metadata, batch split, error propagation |
| `tests/unit/test_preprocessor.py` | ✅ Done | 14 tests: language detection, VTT parsing, dedup, process() full flow, error propagation |
| `tests/unit/test_lambda_handler.py` | ✅ Done | 12 tests: valid EN/HI queries, language detection, 400/500 error paths, direct invocation |

---

## Fixes & Improvements (post-Phase 6)

| Change | File(s) | Notes |
|---|---|---|
| Add `BedrockSynthesisError` | `src/chitrakatha/exceptions.py`, `data/scripts/synthesize_training_pairs.py`, `pipeline/steps/synthesize_pairs.py` | Separate exception for Claude synthesis failures vs. Titan embedding failures; fixed in pipeline step too |
| FAISS index size guard | `src/chitrakatha/ingestion/faiss_writer.py` | Logs warning when index exceeds 50k vectors (cold-start risk threshold) |
| Switch to SageMaker JumpStart (Llama 3.2 3B) | `pipeline/pipeline.py`, `pipeline/steps/train.py` | Replaced `HuggingFace` estimator with `JumpStartEstimator` (`meta-textgeneration-llama-3-2-3b-instruct`); weights fetched via explicit `"model"` training input channel using `model_uris.retrieve()` — no HuggingFace token needed |
| Processing container package fix | `pipeline/pipeline.py` | Added `_make_processing_source_dir()` helper; all `ProcessingStep`s now use `get_run_args(source_dir=...)` to bundle `chitrakatha` package alongside the step script — fixes `ModuleNotFoundError` |
| Training container package fix | `pipeline/pipeline.py` | Added `_make_training_source_dir()` helper; copies full `pipeline/steps/` + `chitrakatha/` package into estimator `source_dir` — fixes `from chitrakatha.monitoring...` in `train.py` |
| Switch to real-time endpoint + scale-to-zero | `serving/deploy_endpoint.py`, `serving/lambda/handler.py` | Replaced `ServerlessInferenceConfig` with real-time `ml.g4dn.xlarge`; App Auto Scaling `MinCapacity=0` enables scale-to-zero; Lambda returns HTTP 503 + `Retry-After: 300` when endpoint is warming up |
| Instance types aligned to 3B model | `pipeline/pipeline.py` | Training + eval both use `ml.g4dn.xlarge` (16 GB VRAM, NVIDIA T4) instead of `ml.g5.2xlarge`; 3B model in 4-bit uses ~2.5 GB VRAM — significant cost reduction per run |
| Container env vars | `pipeline/pipeline.py`, `.github/workflows/ct.yml` | Added `_CONTAINER_ENV` dict injected into all processors so `Settings()` can resolve `S3_VECTORS_BUCKET`, `S3_FAISS_INDEX_PREFIX`, etc.; ct.yml now fetches all required terraform outputs |
| LoRA adapter merge | `pipeline/steps/train.py` | Added `trainer.model.merge_and_unload()` before `save_pretrained()` — without this, serverless endpoint serves unmodified base model |
| inference.py GPU/CPU branch | `serving/inference.py` | GPU path: 4-bit NF4 quantization (~4 GB VRAM); CPU path: bfloat16 with OOM warning (8B model needs real-time GPU endpoint in production) |
| Eval processor instance fix | `pipeline/pipeline.py` | Changed `eval_processor` from `ml.m5.2xlarge` (CPU, hours) to `ml.g5.2xlarge` (GPU, ~20 min) since evaluate.py loads full 8B model |
| Embed/synthesize processor image fix | `pipeline/pipeline.py` | Changed `embed_processor` from GPU Docker image to CPU image — Flow A/B only call Bedrock APIs, no local GPU needed |
