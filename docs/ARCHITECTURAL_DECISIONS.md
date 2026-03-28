# Architectural Decisions — Project Chitrakatha

## Decision Log

| Decision Area | Old / Current | New / Chosen | Reason |
|---|---|---|---|
| **Base LLM (fine-tuning)** | Llama 3.2 3B Instruct (JumpStart) | Qwen2.5-3B-Instruct (HuggingFace Hub) | Better Hindi/Devanagari support, Apache 2.0 (no approval needed), JumpStart was brittle |
| **Base model source** | HuggingFace Hub (external, runtime download) | S3 Gold bucket (`base-models/qwen2.5-3b-instruct/`) | Eliminates external dependency; model cached once, passed as `SM_CHANNEL_MODEL`; faster training start |
| **Training compute** | Managed Spot (`use_spot_instances=True`) | On-demand `ml.g4dn.xlarge` | Spot quota = 0 in this AWS account; on-demand works immediately |
| **Training framework versions** | Unpinned (`>=`) | Exact pins (TRL 0.8.6, transformers 4.40, torch 2.1.0) | TRL 0.29 / transformers 5.x had breaking API changes on every run |
| **RAFT pairs per chunk** | Not specified | 3 Q&A pairs per chunk | Balance between dataset size and synthesis cost |
| **Synthesis model** | Claude 3.5 Sonnet (original) | Claude 3.5 Sonnet v2 (`anthropic.claude-3-5-sonnet-20241022-v2:0`) | Better reasoning, same price tier |
| **Vector store** | FAISS-on-S3 (file, in-RAM) | pgvector on RDS PostgreSQL | Hybrid search (vector + SQL filters), concurrent writes, AWS-native, open source |
| **Networking** | Public (no VPC) | Option B: Private RDS + VPC Endpoints + Custom ECR Docker | Most secure without NAT Gateway cost; ~$48 AUD/month total overhead |
| **VPC setup** | None | New VPC with private subnets; RDS in private subnet | RDS never reachable from public internet |
| **Training container access from VPC** | NAT Gateway (to pull HuggingFace model at runtime) | S3 model cache + custom ECR Docker with deps pre-baked | Model downloaded once to S3 Gold; no HuggingFace runtime dependency; no NAT Gateway ($55 AUD/month saved); Docker rebuilt only when `requirements.txt` changes |
| **Training container** | SageMaker-managed pip install at runtime | Custom Docker image in ECR with all deps pre-baked | Eliminates supply chain risk; reproducible; no runtime PyPI calls |
| **Website chatbot serving** | SageMaker real-time endpoint (fine-tuned Qwen, scale-to-zero GPU) | Bedrock Qwen3 Next 80B A3B (`qwen.qwen3-next-80b-a3b`) + pgvector RAG | GPU scale-to-zero causes 3-5 min cold start — unacceptable for a live website; Bedrock has no cold start, no standing cost. Claude 4.x (Haiku 4.5) requires `ap.` cross-region inference profile not supported in `ap-southeast-2`; Qwen3 on Bedrock works in region, stronger Hindi/Devanagari natively, MoE with 3B active params (cost-efficient). |
| **Fine-tuned Qwen model role** | Production serving (live website) | Benchmarking only (Model Registry) | Qwen is still trained via QLoRA+RAFT pipeline for MLOps learning and quality measurement; not deployed to website |
| **IaC** | Terraform | Terraform (unchanged) | Already established |
| **Frontend** | Lambda + API Gateway | Lambda + API Gateway (unchanged) | Serverless, scale-to-zero |

---

## Pipeline — What Each Flow Produces

```
GitHub Actions triggers SageMaker Pipeline
            │
            ├── Flow A (RAG knowledge base) — powers the website
            │     preprocess → Titan Embed v2 → pgvector
            │     updated every time new raw data is added
            │
            └── Flow B (fine-tuning) — MLOps learning + benchmarking
                  preprocess → Claude 3.5 Sonnet synthesis → QLoRA train Qwen
                  output sits in Model Registry; run ~monthly or when data accumulates
```

Both flows share the same preprocessed Silver JSONL — they diverge after preprocessing.

---

## Serving Architecture

**Website chatbot (always-on, no cold start):**
```
User query → Lambda → Titan Embed → pgvector (top-k chunks) → Bedrock Qwen3 Next 80B A3B → response
```

**Fine-tuned Qwen (benchmarking only, on-demand):**
```
Spin up SageMaker endpoint → run eval queries → compare vs Qwen3 RAG → scale back to 0
```

---

## Networking — Option Comparison (AUD/month, idle baseline)

| Option | Setup | RDS | VPC Endpoints | NAT Gateway | ECR Docker | Total | Security |
|---|---|---|---|---|---|---|---|
| **A** | New VPC, public RDS | ~$20 | — | — | Standard | ~$20 | Low (RDS exposed) |
| **B** ✅ | New VPC, private RDS + VPC endpoints + custom ECR | ~$20 | ~$16 | — | Custom | ~$48 | High |
| **C** | New VPC, private RDS + NAT Gateway | ~$20 | — | ~$55 | Standard | ~$75 | High |
| **D** | Aurora Serverless v2 + VPC endpoints | ~$86+ | ~$16 | — | Custom | ~$114+ | High |

**Selected: Option B** — best security-to-cost ratio; VPC endpoints replace NAT Gateway; custom ECR image pre-bakes all pip dependencies at build time.

---

## Cost Model (AUD/month)

### Standing costs (always running)

| Component | AUD/month |
|---|---|
| RDS PostgreSQL (`db.t4g.micro`) | ~$20 |
| VPC Endpoints (Bedrock + Secrets Manager) | ~$16 |
| KMS CMKs | ~$3 |
| CloudWatch | ~$5 |
| Secrets Manager | ~$1 |
| S3 storage | ~$1 |
| ECR image storage | ~$2 |
| **Standing total** | **~$48** |

### Per pipeline run (when fine-tuning, ~monthly)

| Component | AUD/run |
|---|---|
| Titan Embed (Flow A) | ~$0.10 |
| Claude 3.5 Sonnet synthesis (Flow B) | ~$2.00 |
| SageMaker training (`ml.g4dn.xlarge`, ~45 min) | ~$1.50 |
| SageMaker evaluation | ~$0.50 |
| **Per run total** | **~$4** |

### Per query (website chatbot)

| Volume | AUD/month |
|---|---|
| 500 queries/month | ~$0.50 |
| 2,000 queries/month | ~$2 |
| 10,000 queries/month | ~$10 |

### Monthly total (realistic)

| Scenario | AUD/month |
|---|---|
| Qwen3 RAG only (no fine-tuning) | ~$48 |
| Qwen3 RAG + 1 pipeline run/month | ~$52 |

---

## Fine-tuning Cadence vs RAG Update Cadence

| | RAG (pgvector) | Fine-tuning (Qwen) |
|---|---|---|
| **Trigger** | Every time new raw data is added | ~Monthly, or when benchmark scores degrade |
| **Duration** | Minutes | ~45 min |
| **Cost per run** | ~$0.10 | ~$4 |
| **Effect** | Chatbot immediately knows new content | Improved distractor robustness; measured by ROUGE-L |

---

## Benchmarking — Fine-tuned Qwen2.5-3B vs Qwen3 Next 80B A3B (Bedrock)

| Test | What you measure |
|---|---|
| Domain accuracy | Does fine-tuned Qwen2.5-3B answer Indian comic questions more accurately than Qwen3 RAG? |
| Hindi/Devanagari quality | Does fine-tuning improve Devanagari output over the off-the-shelf Qwen3 base? |
| Distractor robustness | Does fine-tuned Qwen correctly ignore irrelevant retrieved chunks? (RAFT-specific) |
| Hallucination rate | Does fine-tuned Qwen avoid making up facts when answer is not in retrieved context? |
| ROUGE-L score | Automated quality score vs gold Q&A pairs from synthesis step |

---

## Implementation Scope

### Build Now (website-ready foundation)

#### 1. Infrastructure (Terraform)
- Provision RDS PostgreSQL + pgvector extension + VPC (Option B: private subnets + VPC endpoints)
- Enable CORS on API Gateway from day one (ready for any frontend)
- Bedrock Qwen3 Next 80B A3B model enabled in AWS console (one-time manual step)

#### 2. Pipeline — Flow A
- Replace `src/chitrakatha/ingestion/faiss_writer.py` with `pgvector_writer.py` (psycopg2 + pgvector)
- Remove FAISS S3 upload from `pipeline/steps/embed_and_index.py`

#### 3. Serving
- Rewrite `serving/inference.py`: pgvector retrieval + Bedrock Qwen3 Next 80B A3B generation
- Update `serving/lambda/handler.py`: new request/response contract (see below); remove SageMaker endpoint call and 503 cold start logic
- SageMaker real-time endpoint removed from hot path; kept for optional benchmarking only

#### 4. Fine-tuning
- Add `pipeline/Dockerfile` with all `pipeline/requirements.txt` deps pre-baked (custom ECR image)
- Update `pipeline/pipeline.py` estimator to reference ECR image URI
- One-time S3 model cache: `huggingface-cli download Qwen/Qwen2.5-3B-Instruct → aws s3 sync s3://chitrakatha-gold/base-models/qwen2.5-3b-instruct/`
- Update `pipeline/pipeline.py` training step to mount `SM_CHANNEL_MODEL` from S3

### API Contract

```json
// Request
{ "query": "Who created Nagraj?" }

// Response
{ "answer": "Nagraj was created by...", "language": "en", "sources": ["raj_comics_1990.txt"] }
```

`language` is `"hi"` when the query contains Devanagari characters (`[\u0900-\u097F]`), otherwise `"en"`.
`sources` is a deduplicated, sorted list of `source_document` values from retrieved chunks; `[]` if no chunks found.

### Defer (later)

| Item | What it needs |
|---|---|
| Conversation history | DynamoDB session store; add `session_id` + `history` to contract when ready |
| API authentication | API Gateway usage plan + API key or Cognito |
| Rate limiting | API Gateway throttling settings |
| Frontend / chat UI | Static site on S3 + CloudFront, or embed API in existing website |

---

## Cleanup Checklist

Everything below must be removed before or during implementation. Nothing here should exist in the final architecture.

### Terraform — remove resources

- [ ] `infra/terraform/s3.tf` — delete S3 Vectors bucket resource + lifecycle policy + versioning config (remove `prevent_destroy` first, then `terraform destroy` the bucket)
- [ ] `infra/terraform/faiss_index.tf` — delete entire file (only contains the `s3_faiss_index_prefix` local)
- [ ] `infra/terraform/outputs.tf` — remove `s3_vectors_bucket`, `s3_vectors_bucket_arn`, `s3_faiss_index_prefix` outputs
- [ ] `infra/terraform/iam.tf` — remove vectors bucket ARN from `S3ReadWriteProjectBuckets` statement in SageMaker role
- [ ] `infra/terraform/iam.tf` — remove `S3ReadJumpStartPrivateCache` statement (no longer using JumpStart)
- [ ] `infra/terraform/iam.tf` — remove `sagemaker:InvokeEndpoint` from Lambda policy (Lambda calls Bedrock now)
- [ ] `infra/terraform/lambda.tf` — remove `SAGEMAKER_ENDPOINT_NAME` env var from Lambda function

### Code — delete files entirely

- [ ] `src/chitrakatha/ingestion/faiss_writer.py` — replaced by `pgvector_writer.py`
- [ ] `data/scripts/ingest_to_faiss.py` — Flow A orchestration script for FAISS; replaced by pgvector insert in `embed_and_index.py`
- [ ] `tests/unit/test_faiss_writer.py` — replaced by `tests/unit/test_pgvector_writer.py`

### Code — keep but repurpose

- [ ] `serving/deploy_endpoint.py` — remove from live serving path; keep for on-demand benchmarking only (deploy fine-tuned Qwen, compare vs Qwen3 RAG, scale back to 0)

### Code — rewrite in place

- [ ] `serving/inference.py` — Qwen generation → Bedrock Qwen3 Next 80B A3B + pgvector retrieval
- [ ] `serving/lambda/handler.py` — new request/response contract + Bedrock call; remove 503 cold start logic
- [ ] `pipeline/steps/embed_and_index.py` — FAISS S3 upload → pgvector insert
- [ ] `src/chitrakatha/exceptions.py` — rename `S3VectorError` → `PgVectorError`
- [ ] `infra/terraform/faiss_index.tf` → replace with `infra/terraform/pgvector.tf` (RDS locals + pgvector schema init)

### Manual AWS cleanup (one-time, after Terraform changes)

- [ ] Empty S3 Vectors bucket before destroying (Terraform cannot destroy non-empty buckets)
- [ ] Delete SageMaker real-time endpoint if deployed (only needed when benchmarking — spin up on demand)
