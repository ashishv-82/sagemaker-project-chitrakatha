# Project Chitrakatha — Interview Preparation Guide

> Reference for discussing Project Chitrakatha in technical interviews.
> Covers architecture, decisions, trade-offs, security, cost, and lessons learned.

---

## 1. Project Overview & Approach

**Q1. Describe Project Chitrakatha in one paragraph.**

Project Chitrakatha is a production-grade, 100% serverless MLOps platform on AWS that fine-tunes Llama 3.2 3B Instruct into a domain expert on Indian comic book history. It answers questions in both English and Devanagari Hindi. Raw source documents — articles, transcripts, Excel sheets — are dropped into S3, and the pipeline automatically preprocesses them, builds a RAG knowledge base using FAISS-on-S3, generates bilingual RAFT training data using Bedrock Claude, fine-tunes the model with QLoRA on SageMaker Spot instances, evaluates it against three quality thresholds, and registers it to the Model Registry for human approval before deploying to a scale-to-zero real-time endpoint. The entire baseline infrastructure costs ~$5/month when idle.

---

**Q2. Why did you choose RAFT over standard SFT (Supervised Fine-Tuning)?**

Standard SFT trains the model on Q&A pairs in isolation — the model learns facts but not the skill of reading retrieved context and ignoring irrelevant documents. In a RAG system, the model receives retrieved chunks at inference time, some of which will be distractors (irrelevant but topically similar). Without RAFT training, the model may fixate on a distractor and produce a hallucinated answer.

RAFT (Retrieval-Augmented Fine-Tuning) trains on examples that explicitly include a golden document *and* distractor documents, with a chain-of-thought that identifies which document is relevant and why the others are not. This teaches the model the exact skill it needs at inference time: reading retrieved context critically.

---

**Q3. What is the data flow from raw content to a user answer?**

```
Raw files (Bronze S3)
  → Preprocessing: NFC normalise, deduplicate, language-tag, chunk
  → Silver S3 (corpus + training splits)
  → Flow A: Titan Embed v2 → FAISS index on S3 Vectors bucket
  → Flow B: Claude Haiku 4.5 → RAFT Q&A pairs → Gold S3
  → QLoRA fine-tune Llama 3.2 3B on Gold data (Spot)
  → Evaluate (ROUGE-L ≥ 0.35, distractor robustness ≥ 0.70)
  → Model Registry (PendingManualApproval)
  → Human approves → Real-time endpoint deployed
  → Query: embed → FAISS top-k → Llama generates grounded answer
```

---

**Q4. Why did you choose a bilingual (English + Devanagari Hindi) approach?**

Indian comic books like Raj Comics, Amar Chitra Katha, and Durga Comics have a primary audience that reads and thinks in Hindi. A purely English model would exclude a significant portion of that audience. The RAFT training pairs include both `question_en`/`answer_en` and `question_hi`/`answer_hi` fields, so the model is exposed to bilingual examples during fine-tuning. At inference, the Lambda handler detects Devanagari Unicode codepoints (`\u0900–\u097F`) in the query and tags the response language accordingly.

---

**Q5. What was the hardest technical challenge in this project?**

The most persistent challenge was the `source_dir` parameter in SageMaker Pipelines SDK. The pipeline steps needed access to the shared `chitrakatha` Python library, but neither `ScriptProcessor` nor `FrameworkProcessor` subclasses reliably propagate `source_dir` through the `@runnable_by_pipeline` decorator replay phase in SDK ≥ 2.200. The fix required redesigning the package delivery mechanism: upload `src/chitrakatha/` to S3 once in `main()`, mount it via `ProcessingInput` at `/opt/ml/processing/input/src`, and set `PYTHONPATH` in the container environment — the canonical SageMaker pattern for sharing library code across pipeline steps.

---

## 2. Architecture & Design Decisions

**Q6. Why FAISS-on-S3 instead of a managed vector database like Pinecone or OpenSearch?**

Three reasons:
1. **Scale-to-zero cost model**: A managed vector DB like Pinecone has a standing cost (~$70/month minimum). FAISS-on-S3 costs $0 when idle — the index is just a file.
2. **Serverless compatibility**: The endpoint can download the index at cold-start and cache it in RAM. Managed DBs require persistent network connections from a running process.
3. **Sufficient scale**: For a domain-specific corpus of tens of thousands of vectors, FAISS `IndexFlatIP` with L2-normalised vectors (cosine similarity) performs exact nearest-neighbour search in milliseconds from RAM. A managed DB adds network latency for no benefit at this scale.

The trade-off: FAISS-on-S3 is not suitable for a corpus of millions of vectors (index would be too large to load into RAM) or for write-heavy workloads (updating requires download → append → upload).

---

**Q7. Why did you use a three-tier S3 data lake (Bronze / Silver / Gold)?**

The medallion architecture enforces data quality progression:
- **Bronze**: Raw, immutable. Never modified after upload. Full lineage from source.
- **Silver**: Cleaned, normalised, deduplicated JSONL. UTF-8 validated, Devanagari preserved. Safe for model training and embedding.
- **Gold**: Training-ready. RAFT pairs with chain-of-thought, model artifacts, evaluation results, checkpoints.

This means if a preprocessing bug is discovered, you can re-run from Bronze without re-uploading raw data. If synthesis logic changes, you can re-run from Silver without re-preprocessing. Each tier is independently reproducible.

---

**Q8. Why Llama 3.2 3B and not a larger model like Llama 3.1 8B?**

Two reasons:
1. **Cost**: 3B in 4-bit QLoRA fits on `ml.g4dn.xlarge` (16GB VRAM, ~$0.22/hr Spot). An 8B model requires `ml.g5.2xlarge` (24GB VRAM, ~$1.21/hr Spot) — 5× more expensive.
2. **Inference**: The 3B model also fits comfortably on a single GPU at the real-time endpoint. For a domain-specific task with good training data, a smaller well-tuned model outperforms a larger untrained one.

---

**Q9. Why QLoRA instead of full fine-tuning?**

Full fine-tuning of a 3B model requires all parameters in full precision — approximately 12GB just for weights, plus activations and gradients. This demands expensive GPU instances.

QLoRA (Quantised Low-Rank Adaptation) loads the base model in 4-bit NF4 quantisation (~2.5GB), then trains only a small set of low-rank adapter matrices added to the attention layers. The adapter is ~50MB. This means:
- Same `ml.g4dn.xlarge` for both training and inference
- Training time drops from hours to ~30–45 minutes
- Quality loss is minimal for domain-specific tasks with a well-structured training set

---

**Q10. Explain the scale-to-zero serving architecture.**

The real-time endpoint uses App Auto Scaling with `MinCapacity=0`. When no requests arrive for a sustained period, SageMaker scales the endpoint down to zero instances — no GPU running, no cost. Two scaling policies are configured:
1. **Target tracking on `SageMakerVariantInvocationsPerInstance`**: Scale in/out based on throughput
2. **Step scaling on `HasBacklogWithoutCapacity`**: Wake the endpoint when a request arrives with no instances running (cold start)

Cold start takes ~3–5 minutes (JumpStart model download + container init). The Lambda handler detects cold-start by catching `ModelNotReadyException` and returns HTTP 503 with `Retry-After: 300` so the client knows to retry.

---

**Q11. Why did you choose SageMaker JumpStart over HuggingFace Hub for the base model?**

JumpStart delivers model weights from an AWS-managed S3 bucket (`jumpstart-cache-prod-{region}`) directly to the training container over AWS backbone — no internet egress, no HuggingFace token required, and the weights are served from the same region as the training job (low latency). For a production MLOps setup, removing the dependency on an external service (HuggingFace Hub) reduces failure modes and keeps everything within the AWS trust boundary.

---

**Q12. Why use SageMaker Pipelines instead of Airflow, Prefect, or Step Functions?**

SageMaker Pipelines is native to the platform — it provides automatic lineage tracking, integration with the Model Registry, Experiments, and Studio out of the box. The DAG is defined in Python using the same SDK used to submit jobs, so there's no separate orchestration layer to operate. Step Functions would require manual integration with every SageMaker API. Airflow/Prefect adds infrastructure overhead (persistent scheduler). For a pure ML workload, SageMaker Pipelines is the lowest-friction choice.

---

**Q13. How does the pipeline handle Spot instance interruptions during training?**

`train.py` configures SageMaker Managed Spot Training with `checkpoint_s3_uri` pointing to `s3://gold/.../checkpoints/`. The `SFTTrainer` saves checkpoints at regular intervals. If the Spot instance is reclaimed, SageMaker automatically provisions a new instance and resumes from the last checkpoint. `max_wait=86400` (24 hours) gives SageMaker up to 24 hours to acquire a Spot instance if none is immediately available.

---

## 3. Data Processing & Chunking

**Q14. How does the sliding-window chunker work and why 15% overlap?**

The chunker splits text into windows of 512 tokens with a 15% overlap (~77 tokens). The overlap ensures that sentences or concepts spanning a chunk boundary appear in both adjacent chunks — so neither retrieval nor training loses context at the edges. 15% is a common heuristic: below 10% risks losing context; above 20% produces redundant vectors and bloats the index.

---

**Q15. How do you handle Devanagari text throughout the pipeline?**

Every text operation uses NFC Unicode normalisation (`unicodedata.normalize('NFC', text)`), which ensures Devanagari characters are in canonical composed form. No ASCII stripping or encoding coercion is applied anywhere. JSON is serialised with `ensure_ascii=False` so Hindi text is stored as-is rather than as `\uXXXX` escape sequences. The chunker operates on token counts, not byte lengths, so multi-byte Devanagari characters are treated correctly.

---

**Q16. How is deduplication handled?**

SHA-256 content hashing is applied to each cleaned chunk. Before writing a chunk to Silver, its hash is checked against a set of already-seen hashes in the current run. Duplicate chunks (identical content from multiple source files) are silently dropped. SHA-256 is used rather than MD5 (used by the upload script for lineage metadata) because MD5 is not collision-resistant enough for a deduplication decision.

---

## 4. RAG & Retrieval

**Q17. What similarity metric does FAISS use and why?**

`IndexFlatIP` (inner product / dot product). The vectors are L2-normalised before insertion, which means dot product equals cosine similarity. Cosine similarity is appropriate here because it measures directional similarity (semantic meaning) rather than magnitude — a short chunk and a long chunk about the same topic should score equally, not be penalised by length.

---

**Q18. How many chunks are retrieved at query time and why?**

Top-5 chunks. This is a balance between context richness (more chunks = more information for the model) and context window usage (more chunks = longer prompt = slower inference). For a 3B model with a 4K context window, 5 chunks of ~512 tokens each (~2560 tokens) leaves room for the query and generated answer.

---

**Q19. Is the FAISS index updated incrementally or rebuilt from scratch?**

Incrementally. `faiss_writer.py` implements the following pattern:
1. Download existing `index.faiss` + `metadata.pkl` from S3 (or create empty if first run)
2. Load existing vector IDs from metadata
3. Skip any chunk already in the index (idempotency)
4. Append only new vectors
5. Upload updated index back to S3

This means adding a new document to Bronze and re-running the pipeline only adds the new document's vectors without re-embedding the entire corpus.

---

## 5. Model Evaluation

**Q20. What are the three evaluation suites and their thresholds?**

1. **Factual accuracy**: ROUGE-L ≥ 0.35, BERTScore (multilingual), exact-match@1. Tests whether the model produces answers that share significant word overlap with ground truth.
2. **Cross-lingual retrieval**: English query must produce a correct Devanagari answer and vice versa. Tests bilingual generalisation.
3. **Distractor robustness**: Model is given 1 golden chunk + 4 distractor chunks never seen during training. Must still produce the correct answer. Threshold: ≥ 0.70. This is the RAFT-specific test — it validates that the training skill generalised beyond the training distractors.

Pipeline only proceeds to model registration if **both** `rouge_l ≥ 0.35` AND `distractor_robustness ≥ 0.70`.

---

**Q21. Why is distractor robustness more important than ROUGE-L for this project?**

ROUGE-L measures lexical overlap — a model can score well on ROUGE-L by memorising training phrases. Distractor robustness tests a genuine reasoning skill: reading multiple retrieved documents, identifying which one is relevant, and ignoring the others. This is exactly what the production system does at inference time. A model that scores 0.80 ROUGE-L but 0.55 distractor robustness would produce unreliable answers when irrelevant chunks are retrieved — which happens regularly in a real-world corpus.

---

**Q22. How is the held-out evaluation set created?**

During RAFT synthesis (`synthesize_pairs.py`), the generated Q&A pairs are split 90/10 at output time. 90% goes to `s3://gold/.../training-pairs/train/` and 10% to `.../eval/`. The split is done at the file level using a hash of the chunk ID to ensure deterministic, reproducible splits — the same chunk always ends up in the same split regardless of run order.

---

## 6. CI/CD & MLOps

**Q23. What triggers the training pipeline?**

Every push to `main` branch triggers the CT (Continuous Training) workflow in `.github/workflows/ct.yml`. The workflow authenticates to AWS via GitHub OIDC (no long-lived keys), fetches Terraform outputs as environment variables, uploads the `chitrakatha` library to S3, and calls `pipeline.py --execute` which upserts and starts the SageMaker Pipeline. The GitHub Actions job itself finishes in ~3 minutes; the SageMaker pipeline runs for hours independently.

---

**Q24. How does GitHub Actions authenticate to AWS without storing access keys?**

OpenID Connect (OIDC). GitHub acts as an identity provider, issuing a signed JWT for each workflow run. AWS IAM is configured with a trust policy that accepts tokens from `token.actions.githubusercontent.com` scoped to the specific repository (`ashishv-82/sagemaker-project-chitrakatha`). The workflow exchanges this JWT for short-lived AWS credentials using `sts:AssumeRoleWithWebIdentity`. No AWS access keys are stored in GitHub Secrets.

---

**Q25. What is the deployment approval workflow?**

After the pipeline evaluates a model and it passes thresholds, it's registered in SageMaker Model Registry with status `PendingManualApproval`. A human (you) reviews the metrics in Studio, then clicks Approve. The `deploy.yml` workflow is then triggered to run `deploy_endpoint.py`, which picks up the latest approved model package and deploys it to the real-time endpoint. This human-in-the-loop gate prevents an automatically trained model from going live without review.

---

**Q26. How do you ensure the pipeline is idempotent?**

Several layers:
- **Preprocessing**: SHA-256 dedup prevents duplicate chunks in Silver
- **Embedding**: Vector IDs are checked before insertion; re-running embed_and_index skips already-indexed chunks
- **Pipeline upsert**: `pipeline.upsert()` creates the pipeline on first run, updates the definition on subsequent runs — it never creates duplicate pipelines
- **Checkpointing**: Training resumes from the last checkpoint on Spot interruption, not from scratch

---

## 7. Security

**Q27. How is data encrypted at rest and in transit?**

At rest: all four S3 buckets (Bronze, Silver, Gold, Vectors) use AWS KMS Customer Managed Keys (CMKs). Every `PutObject` call requires `kms:GenerateDataKey`; every `GetObject` requires `kms:Decrypt`. The CMK has annual automatic rotation enabled.

In transit: all communication happens over HTTPS/TLS within the AWS network. Processing jobs, training jobs, and endpoint invocations communicate with S3 over AWS backbone — no public internet traversal for data.

---

**Q28. How are secrets managed?**

AWS Secrets Manager. The Bedrock API key placeholder is stored at `chitrakatha/synthetic_data_api_key`. No credentials appear in code, environment variable defaults, or Git history. The SageMaker execution role has `secretsmanager:GetSecretValue` scoped to the `chitrakatha/*` prefix only. Terraform provisions the secret with a placeholder value and `ignore_changes = [secret_string]` so the real value can be injected without Terraform overwriting it.

---

**Q29. How is IAM least-privilege enforced?**

Every role has only the permissions it needs:
- **SageMaker execution role**: S3 access scoped to the 4 named buckets, Bedrock scoped to 2 model ARNs, KMS scoped to the project CMK ARN, Secrets Manager scoped to `chitrakatha/*`
- **Lambda execution role**: `sagemaker:InvokeEndpoint` on the one named endpoint only, CloudWatch Logs on its own log group
- **GitHub Actions role**: SageMaker pipeline actions on `chitrakatha-*` resources only, S3 on the 3 project buckets, no ability to access other AWS resources

No `Resource: "*"` on data-plane actions. The only wildcard resources are on control-plane actions (like App Auto Scaling) where AWS does not support resource-level restrictions.

---

**Q30. What would you add for production-grade security hardening?**

1. **VPC endpoints** for S3, Bedrock, and SageMaker so traffic never traverses the public internet (currently uses PublicInternetOnly for Studio to save ~$50/month in NAT costs)
2. **S3 bucket policies** denying `s3:PutObject` without the KMS CMK header — prevents accidental unencrypted uploads
3. **CloudTrail** for full API audit logging
4. **AWS Config rules** to detect IAM policy drift
5. **SageMaker network isolation** on processing/training jobs — prevents jobs from making outbound internet calls

---

## 8. Cost & Scalability

**Q31. What is the baseline monthly cost and what drives it?**

~$5/month when idle:
- S3 storage: ~$0.50 (versioned buckets)
- KMS CMK: ~$1.00 (key + API calls)
- CloudWatch: ~$3.00 (3 alarms + dashboard)
- Secrets Manager: ~$0.40

The endpoint costs $0 when idle (scale-to-zero). Each pipeline run costs ~$2–4 in Bedrock + training compute. The system is designed so you only pay when you're actively training or serving.

---

**Q32. How does the system scale to handle a much larger corpus?**

Current bottlenecks and solutions:
- **FAISS index size**: `IndexFlatIP` is exact search but scales O(n) with corpus size. At 50k+ vectors (~200MB), cold-start download time becomes noticeable. Solution: switch to `IndexIVFFlat` (approximate search, sublinear lookup) or migrate to a managed service like OpenSearch Serverless
- **Embedding**: Currently sequential Bedrock calls. Solution: parallelize with `concurrent.futures.ThreadPoolExecutor` up to Bedrock's concurrency quota
- **Synthesis**: Claude processes one chunk at a time. Solution: batch with async Bedrock calls
- **Endpoint concurrency**: `MaxConcurrency=5` on the serverless config. For higher traffic, switch to a provisioned real-time endpoint with Auto Scaling

---

**Q33. What is the cost per pipeline run?**

Approximate costs for a medium corpus (~100 source chunks):
- Preprocessing (ml.m5.xlarge, ~5 min): ~$0.05
- Embedding (ml.m5.xlarge, ~10 min, Bedrock Titan): ~$0.15
- Synthesis (ml.m5.xlarge, ~20 min, Claude Haiku 4.5): ~$0.30
- Training (ml.g4dn.xlarge Spot, ~45 min): ~$0.17
- Evaluation (ml.g4dn.xlarge, ~10 min): ~$0.04
- **Total per run: ~$0.70**

---

## 9. Reliability & Resilience

**Q34. What happens if a processing step fails partway through?**

SageMaker Pipelines retries failed steps based on the retry policy configured in the pipeline definition. If the step fails after exhausting retries, the pipeline execution is marked Failed. The step's CloudWatch logs are available in Studio. Because each step writes to S3 atomically and downstream steps only read from S3, a failed step does not corrupt any existing data. The pipeline can be re-triggered and will re-run from the failed step (or from scratch depending on configuration).

---

**Q35. How do you handle Bedrock API throttling during synthesis?**

`embedder.py` implements exponential backoff: on `ThrottlingException` or `ServiceUnavailableException`, it retries up to 3 times with delays of 1s, 2s, 4s. If all retries are exhausted, it raises `BedrockEmbeddingError`. In `synthesize_pairs.py`, per-chunk failures are caught and logged; the step continues processing remaining chunks and reports a count of synthesis errors. This means a throttled chunk doesn't abort the entire synthesis run.

---

**Q36. How does the Lambda handle a cold-starting endpoint?**

When SageMaker scales the endpoint from zero, the first invocation triggers provisioning (~3–5 minutes). If the Lambda receives a request during this period, it catches `ModelNotReadyException` and returns:
```json
HTTP 503
{"error": "Endpoint warming up", "retry_after": 300}
Retry-After: 300
```
The client is instructed to retry after 5 minutes. This is standard HTTP semantics for temporary unavailability.

---

## 10. AI Governance & Responsible AI

**Q37. What guardrails are in place for the generated training data?**

1. **Grounding constraint in the synthesis prompt**: Claude is explicitly instructed to ground every answer strictly in the golden chunk only — no external knowledge, no inference beyond what the document states
2. **Chain-of-thought validation**: Each Q&A pair includes a chain-of-thought that identifies the source document — this makes hallucinated answers detectable during manual review
3. **Human approval gate**: No model reaches production without a human reviewing evaluation metrics in the Model Registry

---

**Q38. How do you prevent the model from hallucinating in production?**

At inference time in `inference.py`:
1. The RAG retrieval step always runs before generation — the model never answers from parametric memory alone
2. The prompt explicitly frames the retrieved chunks as the only authorised knowledge sources
3. If FAISS returns empty results (no relevant chunks found), `inference.py` raises `SageMakerPipelineError` rather than allowing the model to answer from memory
4. The distractor robustness evaluation threshold (≥ 0.70) ensures the deployed model has demonstrated the ability to cite sources correctly rather than confabulating

---

**Q39. What is your approach to model versioning and rollback?**

SageMaker Model Registry maintains all model versions with their evaluation metrics. Each registered model package includes:
- The S3 URI of the model artifacts
- The evaluation JSON (ROUGE-L, distractor robustness scores)
- The pipeline execution ARN (full lineage back to the training data)

If a newly deployed model performs poorly, you can approve the previous version in the Registry and re-run `deploy_endpoint.py` — it always deploys the latest approved version. The endpoint update is in-place with zero downtime (SageMaker shifts traffic to the new model only after health checks pass).

---

**Q40. How do you track data lineage?**

SageMaker native lineage tracking connects: `DataSet → ProcessingJob → TrainingJob → Model → Endpoint`. `lineage.py` wraps the `sagemaker.lineage` APIs to explicitly record this chain. Additionally:
- Every Bronze object has an MD5 checksum in S3 metadata
- Silver JSONL records carry `source_document` field pointing back to the Bronze file
- Gold RAFT pairs carry `source_chunk_id` pointing back to the Silver chunk
- Model artifacts are registered with the pipeline execution ARN

---

## 11. Testing

**Q41. What unit tests cover and what they mock?**

Unit tests use `moto` to mock AWS services (S3, Bedrock, Secrets Manager) — no real AWS calls are made:
- `test_chunker.py`: sliding window logic, Devanagari preservation, overlap correctness, edge cases (empty text, single token)
- `test_embedder.py`: batching logic (groups of 25), dimension check, retry behaviour, Devanagari passthrough
- `test_faiss_writer.py`: idempotency (re-running doesn't duplicate vectors), metadata correctness, batch splitting, error propagation
- `test_preprocessor.py`: language detection, VTT timestamp stripping, SHA-256 dedup, full `process()` flow
- `test_lambda_handler.py`: Pydantic validation, language detection (Devanagari/English), HTTP 400/500 paths, direct invocation

Coverage threshold: 80% (enforced in `ci.yml`).

---

**Q42. What integration tests exist?**

`tests/integration/test_pipeline_dag.py` validates the SageMaker Pipeline DAG structure without making AWS calls — it checks that all steps are connected correctly, that the ConditionStep has the right threshold expressions, and that no hardcoded ARNs exist in the pipeline definition. This catches structural bugs before a pipeline run is triggered.

---

**Q43. How would you test the RAG quality end-to-end?**

Beyond the automated evaluation suites, you would:
1. Maintain a curated golden set of 50–100 domain-specific questions with ground-truth answers, updated as the corpus grows
2. Run this golden set against every newly deployed model version
3. Track metric trends in SageMaker Experiments across pipeline runs — a consistently declining ROUGE-L trend signals corpus quality degradation
4. Periodically run adversarial queries (questions about topics not in the corpus) to verify the model declines to answer rather than hallucinating

---

## 12. Common Issues & Lessons Learned

**Q44. What IAM permissions were most commonly missing and why?**

Several were discovered only through live execution:
- `sagemaker:ListTags` — needed by `pipeline.upsert()` on the second run when the pipeline already exists and it merges existing tags
- `sagemaker:AddTags` on `pipelines-*` resources — SageMaker auto-tags every ProcessingJob/TrainingJob with pipeline metadata, requiring the execution role to have this on the auto-generated job names
- `iam:PassRole` (self-pass) — the pipeline orchestrator must pass the execution role to each child job it creates
- `aws-marketplace:ViewSubscriptions` — required for Anthropic models on Bedrock even after account-level subscription approval
- KMS `GenerateDataKey` for the GitHub Actions role — needed to write to KMS-encrypted S3 buckets during the `_sync_chitrakatha_to_s3()` step

---

**Q45. What was the `source_dir` issue and how did you resolve it?**

The SageMaker SDK's `ProcessingStep` requires `step_args=processor.run(...)` rather than `processor.get_run_args()`. When `run()` is called in a `PipelineSession` context, the `@runnable_by_pipeline` decorator intercepts it and returns `RunArgs` instead of submitting a job. However, neither `ScriptProcessor.run()` nor `FrameworkProcessor.run()` reliably propagates the `source_dir` parameter through this replay mechanism in SDK ≥ 2.200.

The solution was to abandon `source_dir` entirely and use the S3 + ProcessingInput pattern: upload the library to S3 once per run, mount it in every container via `ProcessingInput`, and set `PYTHONPATH` in the container environment. This is the documented canonical approach for sharing code across pipeline steps.

---

**Q46. Why did Claude Haiku 4.5 require a different invocation approach than earlier models?**

Claude 4.x models on AWS Bedrock are not available via direct on-demand invocation by model ID. They must be called through a **cross-region inference profile**, which has the format `{region-prefix}.{model-id}` (e.g. `ap.anthropic.claude-haiku-4-5-20251001-v1:0` for AP regions). This profile routes requests through AWS's multi-region inference fleet. The IAM policy must also allow the inference profile ARN (`arn:aws:bedrock:*::inference-profile/ap.anthropic...`), not just the foundation model ARN.

---

## 13. Architecture Trade-offs & Alternative Approaches

**Q47. What are the trade-offs of FAISS-on-S3 compared to a managed vector database?**

| Aspect | FAISS-on-S3 | Managed Vector DB (Pinecone/OpenSearch) |
|---|---|---|
| Cost at rest | $0 | $70–200/month |
| Cold-start | 2–5s download | Near-instant |
| Concurrent writes | Single-writer (download→append→upload) | Native concurrent writes |
| Scale | Up to ~50k vectors comfortably | Millions of vectors |
| Consistency | Eventual (S3 object replacement) | Strong |
| Ops burden | Zero (just S3) | Service to manage/monitor |

For this project's scale and cost constraints, FAISS-on-S3 is the correct choice. At 500k+ vectors or high write concurrency, a managed service becomes justified.

---

**Q48. Why not use LangChain or LangGraph?**

This project's RAG flow is: embed query → FAISS top-k → generate. That's 3 lines of code. LangChain/LangGraph solve a different problem: multi-step agent loops where an LLM decides what action to take next, calls tools, and iterates. Adding LangChain here would introduce ~150MB of dependencies, abstract away code we need to understand and debug, and add cold-start latency — for zero benefit. Complexity should match the problem.

---

**Q49. What would you change if you had to handle 10× the current data volume?**

1. Replace `IndexFlatIP` with `IndexIVFFlat` (approximate search, sublinear cost) with periodic index rebuilds
2. Parallelize Bedrock embedding calls using `asyncio` / `ThreadPoolExecutor`
3. Partition the Bronze bucket by date and source to enable incremental processing (only re-process new files)
4. Move from a single ScriptProcessor for synthesis to a distributed processing pattern (multiple containers processing different corpus partitions)
5. Add a DynamoDB table to track processed chunk IDs as a more scalable alternative to loading the full FAISS metadata into memory for idempotency checks

---

**Q50. If you were to add multimodal support (comic book images), what would change?**

1. **Ingestion**: Add an image parser to `preprocessing.py` (PyMuPDF or Pillow for panel extraction)
2. **Embedding**: Replace Titan Embed Text v2 with a multimodal embedding model (e.g. Amazon Titan Multimodal Embeddings G1)
3. **FAISS index**: Maintain two separate indices (text, image) or a combined multimodal index — dimension would change from 1024 to 1408
4. **Training data**: Extend the RAFT synthesis prompt to include image captions alongside text chunks
5. **Serving**: Update `inference.py` to handle image input and multimodal context assembly
6. **IAM**: Add the Titan Multimodal model ARN to the Bedrock policy

The core pipeline DAG structure and serving architecture would remain unchanged.

---

## Bonus: Quick-fire Questions

**Q51. What does NFC normalisation do?**
Converts Unicode characters to Canonical Decomposition followed by Canonical Composition — ensures Devanagari characters are in a consistent composed form so the same word is always represented identically regardless of how it was typed.

**Q52. Why does the FAISS index use `IndexFlatIP` and not `IndexFlatL2`?**
`IndexFlatL2` minimises Euclidean distance, which is magnitude-sensitive. `IndexFlatIP` (inner product) on L2-normalised vectors computes cosine similarity, which is magnitude-invariant and better captures semantic similarity between text passages of varying length.

**Q53. What is the LoRA rank and why was r=16 chosen?**
LoRA rank `r=16` means the weight update matrices have rank 16, adding ~8M trainable parameters to the frozen 3B base model (~0.3% of total parameters). `r=16` is a standard starting point — high enough to capture domain-specific knowledge, low enough to train quickly. `r=32` would improve quality at the cost of 2× training time.

**Q54. What is `lora_alpha` and how does it relate to rank?**
`lora_alpha=32` is the LoRA scaling factor. The effective scaling applied to the adapter is `alpha/rank = 32/16 = 2.0`. This controls how strongly the adapter updates influence the base model. Setting `alpha = 2 × rank` is a common heuristic that works well across tasks.

**Q55. How would you detect and handle data drift in production?**
Monitor the distribution of FAISS similarity scores for production queries — if average top-1 similarity drops significantly, it suggests the corpus no longer covers the topics users are asking about (data drift). CloudWatch alarms on the custom metric `faiss_top1_similarity` would trigger a notification. The response would be to add new source documents to Bronze and re-run the pipeline to update both the FAISS index and retrain the model.
