"""SageMaker Pipeline DAG — Project Chitrakatha.

Why: Defining the pipeline as a ``sagemaker.workflow.pipeline.Pipeline`` DAG
     rather than a series of ad-hoc scripts provides:
       - CI/CD triggering: one `pipeline.upsert() + execution.start()` call
         from GitHub Actions, no manual steps.
       - Automatic parallelism: Flow A (embed_and_index) and Flow B
         (synthesize_pairs) run concurrently — no dependency between them.
       - Conditional registration: the model is only registered if eval passes
         ROUGE-L >= 0.35 AND distractor_robustness >= 0.70.
       - Lineage: SageMaker automatically tracks every step's inputs/outputs.

Pipeline steps (in execution order):
    1. preprocessing      — Bronze → Silver (corpus + training split)
    2a. embed_and_index   ┐ parallel
    2b. synthesize_pairs  ┘
    3.  train             — QLoRA + RAFT on Gold Q&A
    4.  evaluate          — 3-suite evaluation
    5.  condition         — ROUGE-L >= 0.35 AND distractor_robustness >= 0.70
    6.  create_model      — Package model artifact
    7.  register_model    — SageMaker Model Registry (PendingManualApproval)

Pipeline parameters (overridable per execution):
    InputDataUri    — S3 Bronze URI for this pipeline run
    ModelApprovalStatus — default: PendingManualApproval
    BaseModelId     — HuggingFace model ID for the base LLM

Constraints:
    - All ARNs and bucket names from environment (Terraform outputs) — no hardcoding.
    - Tags: Project: Chitrakatha, CostCenter: MLOps-Research on every resource.
    - Spot training: train_use_spot_instances=True, max_wait=86400.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.huggingface import HuggingFace
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput, ScriptProcessor
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

logger = logging.getLogger(__name__)

# All config from environment — populated by GitHub Actions from `terraform output`.
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]
AWS_REGION = os.environ.get("AWS_REGION", "ap-southeast-2")
BRONZE_BUCKET = os.environ["BRONZE_BUCKET"]
SILVER_BUCKET = os.environ["SILVER_BUCKET"]
GOLD_BUCKET = os.environ["GOLD_BUCKET"]
KMS_KEY_ARN = os.environ["KMS_KEY_ARN"]
PIPELINE_NAME = "chitrakatha-mlops-pipeline"

# Source directory for pipeline step scripts (relative to this file).
STEPS_DIR = Path(__file__).parent / "steps"

# Resource tags applied to every SageMaker resource created by this pipeline.
RESOURCE_TAGS = [
    {"Key": "Project", "Value": "Chitrakatha"},
    {"Key": "CostCenter", "Value": "MLOps-Research"},
]


def create_pipeline(session: PipelineSession | None = None) -> Pipeline:
    """Build and return the Chitrakatha SageMaker Pipeline definition.

    Args:
        session: Optional ``PipelineSession`` (injected for testing).

    Returns:
        Configured ``Pipeline`` object (not yet upserted to SageMaker).
    """
    sm_session = session or PipelineSession(default_bucket=GOLD_BUCKET)

    ###########################################################################
    # Pipeline Parameters
    ###########################################################################

    input_data_uri = ParameterString(
        name="InputDataUri",
        default_value=f"s3://{BRONZE_BUCKET}/",
    )
    model_approval_status = ParameterString(
        name="ModelApprovalStatus",
        default_value="PendingManualApproval",
    )
    base_model_id = ParameterString(
        name="BaseModelId",
        default_value="meta-llama/Meta-Llama-3.1-8B-Instruct",
    )

    ###########################################################################
    # Step 1: Preprocessing — Bronze → Silver
    ###########################################################################

    sklearn_processor = SKLearnProcessor(
        framework_version="1.2-1",
        role=ROLE_ARN,
        instance_type="ml.m5.xlarge",
        instance_count=1,
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
    )

    step_preprocessing = ProcessingStep(
        name="Preprocessing",
        processor=sklearn_processor,
        inputs=[
            ProcessingInput(
                source=input_data_uri,
                destination="/opt/ml/processing/input/bronze",
                input_name="bronze",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="corpus",
                source="/opt/ml/processing/output/corpus",
                destination=f"s3://{SILVER_BUCKET}/corpus/",
            ),
            ProcessingOutput(
                output_name="training",
                source="/opt/ml/processing/output/training",
                destination=f"s3://{SILVER_BUCKET}/training/",
            ),
        ],
        code=str(STEPS_DIR / "preprocessing.py"),
        job_arguments=["--output-data-dir", "/opt/ml/processing/output"],
    )

    ###########################################################################
    # Step 2a: Embed & Index — Flow A (runs in parallel with Step 2b)
    ###########################################################################

    embed_processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
        env={"SAGEMAKER_EXPERIMENT_RUN": "chitrakatha-pipeline-run"},
    )

    step_embed = ProcessingStep(
        name="EmbedAndIndex",
        processor=embed_processor,
        inputs=[
            ProcessingInput(
                source=step_preprocessing.properties.ProcessingOutputConfig.Outputs["corpus"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/corpus",
                input_name="corpus",
            )
        ],
        code=str(STEPS_DIR / "embed_and_index.py"),
        depends_on=[step_preprocessing],
    )

    ###########################################################################
    # Step 2b: Synthesize Pairs — Flow B (parallel with Step 2a)
    ###########################################################################

    step_synthesize = ProcessingStep(
        name="SynthesizePairs",
        processor=embed_processor,
        inputs=[
            ProcessingInput(
                source=step_preprocessing.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/training",
                input_name="training",
            )
        ],
        outputs=[
            ProcessingOutput(
                output_name="gold",
                source="/opt/ml/processing/output/gold",
                destination=f"s3://{GOLD_BUCKET}/training-pairs/",
            )
        ],
        code=str(STEPS_DIR / "synthesize_pairs.py"),
        depends_on=[step_preprocessing],
    )

    ###########################################################################
    # Step 3: Training — QLoRA + RAFT (Spot instance)
    ###########################################################################

    huggingface_estimator = HuggingFace(
        entry_point="train.py",
        source_dir=str(STEPS_DIR),
        role=ROLE_ARN,
        instance_type="ml.g5.2xlarge",
        instance_count=1,
        transformers_version="4.36",
        pytorch_version="2.1",
        py_version="py310",
        # Managed Spot Training — up to 70% cost reduction.
        use_spot_instances=True,
        max_wait=86400,  # 24 hours max total wait including spot interruption.
        max_run=7200,    # 2 hours max actual training time.
        checkpoint_s3_uri=f"s3://{GOLD_BUCKET}/checkpoints/",
        hyperparameters={
            "MODEL_ID": base_model_id,
            "EPOCHS": "3",
            "BATCH_SIZE": "4",
            "LEARNING_RATE": "2e-4",
            "LORA_R": "16",
            "LORA_ALPHA": "32",
        },
        environment={"SAGEMAKER_EXPERIMENT_RUN": "chitrakatha-pipeline-run"},
        output_path=f"s3://{GOLD_BUCKET}/model-artifacts/",
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
    )

    step_train = TrainingStep(
        name="TrainQLoRARAFT",
        estimator=huggingface_estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=step_synthesize.properties.ProcessingOutputConfig.Outputs["gold"].S3Output.S3Uri,
                content_type="application/x-ndjson",
            )
        },
        depends_on=[step_embed, step_synthesize],
    )

    ###########################################################################
    # Step 4: Evaluate — 3-suite evaluation
    ###########################################################################

    eval_processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateRAFT",
        processor=eval_processor,
        inputs=[
            ProcessingInput(
                source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/input/model",
                input_name="model",
            ),
            ProcessingInput(
                source=step_synthesize.properties.ProcessingOutputConfig.Outputs["gold"].S3Output.S3Uri,
                destination="/opt/ml/processing/input/eval",
                input_name="eval",
            ),
        ],
        outputs=[
            ProcessingOutput(
                output_name="evaluation",
                source="/opt/ml/processing/output/evaluation",
                destination=f"s3://{GOLD_BUCKET}/evaluation/",
            )
        ],
        code=str(STEPS_DIR / "evaluate.py"),
        property_files=[evaluation_report],
    )

    ###########################################################################
    # Step 5: Condition — dual threshold gate
    ###########################################################################

    cond_rouge = ConditionGreaterThanOrEqualTo(
        left=sagemaker.workflow.functions.JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="rouge_l",
        ),
        right=0.35,
    )

    cond_robustness = ConditionGreaterThanOrEqualTo(
        left=sagemaker.workflow.functions.JsonGet(
            step_name=step_evaluate.name,
            property_file=evaluation_report,
            json_path="distractor_robustness",
        ),
        right=0.70,
    )

    ###########################################################################
    # Step 6: Create Model
    ###########################################################################

    model = Model(
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-inference:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        role=ROLE_ARN,
        entry_point="inference.py",
        source_dir=str(Path(__file__).parent.parent / "serving"),
        sagemaker_session=sm_session,
    )

    step_create_model = ModelStep(
        name="CreateChitrakathaModel",
        step_args=model.create(instance_type="ml.m5.xlarge"),
    )

    ###########################################################################
    # Step 7: Register Model — PendingManualApproval
    ###########################################################################

    step_register = RegisterModel(
        name="RegisterChitrakathaModel",
        estimator=huggingface_estimator,
        model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["application/json"],
        response_types=["application/json"],
        approval_status=model_approval_status,
        model_package_group_name="ChitrakathaModelPackageGroup",
    )

    ###########################################################################
    # Step 5 (condition): only proceed if both thresholds pass
    ###########################################################################

    step_condition = ConditionStep(
        name="CheckEvalThresholds",
        conditions=[cond_rouge, cond_robustness],
        if_steps=[step_create_model, step_register],
        else_steps=[],
    )

    ###########################################################################
    # Assemble Pipeline
    ###########################################################################

    pipeline = Pipeline(
        name=PIPELINE_NAME,
        parameters=[input_data_uri, model_approval_status, base_model_id],
        steps=[
            step_preprocessing,
            step_embed,
            step_synthesize,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
    )

    return pipeline


def main() -> None:
    """CLI entry point: upsert and optionally execute the pipeline."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Chitrakatha SageMaker Pipeline")
    parser.add_argument("--execute", action="store_true", help="Start a pipeline execution after upsert.")
    parser.add_argument(
        "--input-uri",
        default=f"s3://{os.environ.get('BRONZE_BUCKET', '')}/",
        help="S3 URI for Bronze input data.",
    )
    args = parser.parse_args()

    pipeline = create_pipeline()
    pipeline.upsert(role_arn=ROLE_ARN, tags=RESOURCE_TAGS)
    logger.info("Pipeline '%s' upserted successfully.", PIPELINE_NAME)

    if args.execute:
        execution = pipeline.start(
            parameters={"InputDataUri": args.input_uri}
        )
        logger.info("Pipeline execution started: %s", execution.arn)
        print(f"PIPELINE_EXECUTION_ARN={execution.arn}")


if __name__ == "__main__":
    main()
