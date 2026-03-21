# Project Chitrakatha — Progress Tracker

> Last updated: 2026-03-21  
> Reference: [IMPLEMENTATION_PLAN.md](./IMPLEMENTATION_PLAN.md)

---

## Phase Summary

| Phase | Description | Status |
|---|---|---|
| **Phase 0** | Repo scaffold & governance | ✅ Complete |
| **Phase 1** | Terraform IaC | ✅ Complete |
| **Phase 2** | Data layer & ingestion | 🔲 Not started |
| **Phase 3** | SageMaker MLOps pipeline | 🔲 Not started |
| **Phase 4** | Serving & Lambda bridge | 🔲 Not started |
| **Phase 5** | Observability & lineage | 🔲 Not started |
| **Phase 6** | CI/CD (GitHub Actions) | 🔲 Not started |

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

## Phase 2 — Data Layer & Ingestion 🔲

| File | Status | Notes |
|---|---|---|
| `data/scripts/upload_to_bronze.py` | 🔲 | UTF-8 validated upload; MD5 checksum in metadata |
| `src/chitrakatha/ingestion/chunker.py` | 🔲 | Sliding-window, 15% overlap, Devanagari-safe |
| `src/chitrakatha/ingestion/embedder.py` | 🔲 | Bedrock Titan Embed v2, batch 25 |
| `src/chitrakatha/ingestion/vector_writer.py` | 🔲 | S3 Vectors writer |
| `data/scripts/ingest_to_vectors.py` | 🔲 | Idempotent orchestration (Flow A) |
| `data/scripts/synthesize_training_pairs.py` | 🔲 | **RAFT**: Claude 3.5 Sonnet → golden + 2 distractors + CoT + bilingual answer |

---

## Phase 3 — SageMaker MLOps Pipeline 🔲

| File | Status | Notes |
|---|---|---|
| `pipeline/steps/preprocessing.py` | 🔲 | Bronze → Silver (corpus + training split) |
| `pipeline/steps/embed_and_index.py` | 🔲 | Flow A: corpus → S3 Vectors |
| `pipeline/steps/synthesize_pairs.py` | 🔲 | **RAFT**: chunks → Claude → Gold JSONL (golden + distractors + CoT) |
| `pipeline/steps/train.py` | 🔲 | QLoRA + **RAFT prompt template** (shuffled docs), Spot, Experiments |
| `pipeline/steps/evaluate.py` | 🔲 | 3 suites: factual accuracy, cross-lingual, **distractor robustness ≥ 0.70** |
| `pipeline/pipeline.py` | 🔲 | Full DAG (8 steps), ConditionStep: ROUGE-L ≥ 0.35 **AND** distractor_robustness ≥ 0.70 |
| `pipeline/requirements.txt` | 🔲 | Training container deps incl. `sentencepiece` (multilingual BERTScore) |

---

## Phase 4 — Serving & Lambda Bridge 🔲

| File | Status | Notes |
|---|---|---|
| `serving/deploy_endpoint.py` | 🔲 | ServerlessInferenceConfig (6144MB, max 5) |
| `serving/inference.py` | 🔲 | RAG predict_fn: embed → S3 Vectors → Llama |
| `serving/lambda/handler.py` | 🔲 | Language-aware, pydantic validation |
| `serving/lambda/requirements.txt` | 🔲 | `boto3`, `pydantic>=2` |
| `infra/terraform/lambda.tf` | 🔲 | Lambda function + API Gateway HTTP API trigger |

---

## Phase 5 — Observability & Lineage 🔲

| File | Status | Notes |
|---|---|---|
| `src/chitrakatha/monitoring/lineage.py` | 🔲 | SageMaker Lineage chain |
| `src/chitrakatha/monitoring/experiments.py` | 🔲 | Hyperparams + eval metrics logger |
| `infra/terraform/cloudwatch.tf` (additions) | 🔲 | Dashboard: invocations, error rate, cold-start |

---

## Phase 6 — CI/CD (GitHub Actions) 🔲

| File | Status | Notes |
|---|---|---|
| `.github/workflows/ci.yml` | 🔲 | Lint, type-check, unit tests, tf-validate |
| `.github/workflows/ct.yml` | 🔲 | Trigger SageMaker Pipeline on merge to main |
| `.github/workflows/deploy.yml` | 🔲 | Auto-deploy on Model Registry approval |

---

## Unit Tests 🔲

| File | Status |
|---|---|
| `tests/unit/test_chunker.py` | 🔲 |
| `tests/unit/test_embedder.py` | 🔲 |
| `tests/unit/test_preprocessor.py` | 🔲 |
| `tests/unit/test_vector_writer.py` | 🔲 |
| `tests/unit/test_lambda_handler.py` | 🔲 |
| `tests/integration/test_pipeline_dag.py` | 🔲 |
