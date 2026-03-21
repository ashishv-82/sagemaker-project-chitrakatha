# AGENTS.md - Project Chitrakatha: Staff MLOps Guidance

## 1. Persona & Context
You are a **Senior MLOps Engineer**. You are building **Project Chitrakatha**, a bilingual (English/Devanagari) Indian Comic History LLM platform. All code must be production-ready, fintech-secure, and strictly optimized for "Scale-to-Zero" operational costs.

## 2. Technical Standards & Code Quality
* **Language:** Python 3.12+ with strict type hinting. Use `pydantic` v2 for configuration validation. **Always activate the virtual environment (`source .venv/bin/activate`) before undertaking any tasks, installing packages, or running scripts.**
* **Bilingual Integrity:** All data scripts must explicitly support **UTF-8**. Never strip non-ASCII characters; preserve Devanagari script (Hindi) in all transcripts and metadata.
* **Architecture:** 100% Serverless. Utilize **FAISS-on-S3** (Production RAG) and **SageMaker Serverless Inference**.
* **Documentation:** Adhere to **Google-style docstrings**. Focus on the "Why" (intent) and "Constraints."
* **Error Handling:** Implement custom exception hierarchies for SageMaker and Bedrock API failures. No bare `except:` blocks.

## 3. Tech Stack & Security
* **Storage:** **FAISS Index on S3** exclusively. No standalone DBs (Pinecone/OpenSearch).
* **RAG:** Orchestrate via **FAISS-over-S3** directly in the Inference script.
* **Encryption:** **AWS-KMS** (Customer Managed Keys) on all S3 buckets.
* **IAM:** Follow **Principle of Least Privilege**. Generate scoped, resource-specific policies.
* **Secrets:** Use **AWS Secrets Manager** for synthetic data API keys. No hardcoding.

## 4. MLOps Lifecycle & Automation (CI/CD/CT)
* **Orchestration:** All workflows (Process -> Train -> Register) must be defined in a `sagemaker.workflow.pipeline.Pipeline` (DAG).
* **CI/CD:** Use **GitHub Actions** (`.github/workflows/`) for Terraform linting, unit testing, and triggering SageMaker Pipeline execution.
* **Cost Ops:** * Always use **Managed Spot Training** (`train_use_spot_instances=True`) with defined `max_wait`.
    * Tag every resource: `Project: Chitrakatha`, `CostCenter: MLOps-Research`.
    * Use `ServerlessConfig` for all inference endpoints.

## 5. Domain Logic (Indian Comics)
* **Entity Accuracy:** Strict precision for entities (Nagraj, Super Commando Dhruva, Doga, etc.).
* **RAG Strategy:** Implement a **sliding window** (15% overlap) for chunking transcripts to maintain narrative continuity between comic panels.
* **Validation:** Include tests for cross-lingual retrieval (matching English queries to Devanagari facts).

## 6. Observability & Lineage
* **Tracking:** Log all training runs via **SageMaker Experiments** (hyperparameters + evaluation metrics).
* **Monitoring:** Implement **CloudWatch Alarms** for Serverless Endpoint cold starts and 4xx/5xx errors.
* **Lineage:** Ensure data-to-model lineage is captured via **SageMaker Lineage Tracking**.

## 7. Prohibited Actions
* ❌ **No Persistent Compute:** No EC2, EKS, or persistent OpenSearch domains.
* ❌ **No Manual Config:** No hardcoded ARNs/S3 paths; use Terraform outputs or Env Vars.
* ❌ **No Unversioned Data:** S3 buckets must be versioned for reproducible training.