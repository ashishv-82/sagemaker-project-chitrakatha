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
    InputDataUri        — S3 Bronze URI for this pipeline run
    ModelApprovalStatus — default: PendingManualApproval

Base model:
    meta-textgeneration-llama-3-2-3b-instruct (SageMaker JumpStart)
    Weights fetched from AWS-managed S3 — no HuggingFace token required.

Constraints:
    - All ARNs and bucket names from environment (Terraform outputs) — no hardcoding.
    - Tags: Project: Chitrakatha, CostCenter: MLOps-Research on every resource.
    - Spot training: train_use_spot_instances=True, max_wait=86400.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import sagemaker
from sagemaker import model_uris
from sagemaker.jumpstart.estimator import JumpStartEstimator
from sagemaker.model import Model
from sagemaker.processing import ProcessingInput, ProcessingOutput
from sagemaker.pytorch import PyTorchProcessor
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
VECTORS_BUCKET = os.environ["VECTORS_BUCKET"]
FAISS_INDEX_PREFIX = os.environ.get("FAISS_INDEX_PREFIX", "faiss-index")

PIPELINE_NAME = "chitrakatha-mlops-pipeline"

# SageMaker JumpStart model ID — weights are fetched from AWS-managed S3,
# no HuggingFace account or token required.
# Llama 3.2 3B: fits on ml.g4dn.xlarge (16 GB VRAM) in 4-bit (~2.5 GB),
# and serves via a real-time GPU endpoint with App Auto Scaling (scale-to-zero).
JUMPSTART_MODEL_ID = "meta-textgeneration-llama-3-2-3b-instruct"

# Source directory for pipeline step scripts (relative to this file).
STEPS_DIR = Path(__file__).parent / "steps"
ROOT_DIR = Path(__file__).parent.parent

# Resource tags applied to every SageMaker resource created by this pipeline.
RESOURCE_TAGS = [
    {"Key": "Project", "Value": "Chitrakatha"},
    {"Key": "CostCenter", "Value": "MLOps-Research"},
]


def _make_processing_source_dir(step_script: Path) -> str:
    """Bundle a step script with the chitrakatha package into a temp directory.

    SageMaker uploads the entire source_dir to S3 and sets it as the working
    directory inside the processing container, so ``import chitrakatha`` resolves
    without requiring a custom Docker image or a pip-install shell command.

    If a companion ``<stem>_requirements.txt`` exists alongside the step script,
    it is copied into the temp dir as ``requirements.txt`` so the processor can
    install extra packages (e.g. openpyxl for the sklearn preprocessing container).

    Args:
        step_script: Absolute path to the pipeline step .py file.

    Returns:
        Path string of the temp directory. SageMaker uploads it during
        ``pipeline.upsert()``, so it must exist until that call returns.
    """
    tmp = Path(tempfile.mkdtemp(prefix="chitrakatha_proc_"))
    shutil.copy2(step_script, tmp / step_script.name)
    shutil.copytree(ROOT_DIR / "src" / "chitrakatha", tmp / "chitrakatha")
    # Copy companion requirements.txt if present (e.g. preprocessing_requirements.txt).
    companion_reqs = step_script.parent / f"{step_script.stem}_requirements.txt"
    if companion_reqs.exists():
        shutil.copy2(companion_reqs, tmp / "requirements.txt")
    return str(tmp)


def _make_training_source_dir() -> str:
    """Bundle all pipeline/steps files with the chitrakatha package.

    Unlike ``_make_processing_source_dir`` (which handles one script at a time),
    the training container needs the full steps directory (requirements.txt,
    train.py, etc.) plus chitrakatha so that ``from chitrakatha.monitoring...``
    resolves inside the training job.
    """
    tmp = Path(tempfile.mkdtemp(prefix="chitrakatha_train_"))
    for item in STEPS_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, tmp / item.name)
    shutil.copytree(ROOT_DIR / "src" / "chitrakatha", tmp / "chitrakatha")
    return str(tmp)


def create_pipeline(session: PipelineSession | None = None) -> Pipeline:
    """Build and return the Chitrakatha SageMaker Pipeline definition.

    Args:
        session: Optional ``PipelineSession`` (injected for testing).

    Returns:
        Configured ``Pipeline`` object (not yet upserted to SageMaker).
    """
    sm_session = session or PipelineSession(default_bucket=GOLD_BUCKET)

    # Injected into every processing container so Settings() can resolve all
    # required fields without hitting the environment of the calling machine.
    _CONTAINER_ENV: dict[str, str] = {
        "S3_BRONZE_BUCKET": BRONZE_BUCKET,
        "S3_SILVER_BUCKET": SILVER_BUCKET,
        "S3_GOLD_BUCKET": GOLD_BUCKET,
        "S3_VECTORS_BUCKET": VECTORS_BUCKET,
        "S3_FAISS_INDEX_PREFIX": FAISS_INDEX_PREFIX,
        "KMS_KEY_ARN": KMS_KEY_ARN,
        "SAGEMAKER_ROLE_ARN": ROLE_ARN,
        "AWS_REGION": AWS_REGION,
    }

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
        env=_CONTAINER_ENV,
    )

    step_preprocessing = ProcessingStep(
        name="Preprocessing",
        step_args=sklearn_processor.run(
            code="preprocessing.py",
            source_dir=_make_processing_source_dir(STEPS_DIR / "preprocessing.py"),
            requirements="requirements.txt",
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
            arguments=["--output-data-dir", "/opt/ml/processing/output"],
        ),
    )

    ###########################################################################
    # Step 2a: Embed & Index — Flow A (runs in parallel with Step 2b)
    ###########################################################################

    # CPU image: Flow A/B only call Bedrock APIs — no GPU computation required.
    # PyTorchProcessor (FrameworkProcessor subclass) is used instead of ScriptProcessor
    # because it supports source_dir in run(), which ScriptProcessor does not.
    embed_processor = PyTorchProcessor(
        framework_version="2.1.0",
        py_version="py310",
        role=ROLE_ARN,
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
        env={**_CONTAINER_ENV, "SAGEMAKER_EXPERIMENT_RUN": "chitrakatha-pipeline-run"},
    )

    step_embed = ProcessingStep(
        name="EmbedAndIndex",
        step_args=embed_processor.run(
            code="embed_and_index.py",
            source_dir=_make_processing_source_dir(STEPS_DIR / "embed_and_index.py"),
            inputs=[
                ProcessingInput(
                    source=step_preprocessing.properties.ProcessingOutputConfig.Outputs["corpus"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/corpus",
                    input_name="corpus",
                )
            ],
        ),
        depends_on=[step_preprocessing],
    )

    ###########################################################################
    # Step 2b: Synthesize Pairs — Flow B (parallel with Step 2a)
    ###########################################################################

    step_synthesize = ProcessingStep(
        name="SynthesizePairs",
        step_args=embed_processor.run(
            code="synthesize_pairs.py",
            source_dir=_make_processing_source_dir(STEPS_DIR / "synthesize_pairs.py"),
            inputs=[
                ProcessingInput(
                    source=step_preprocessing.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/training",
                    input_name="training",
                )
            ],
            outputs=[
                ProcessingOutput(
                    output_name="gold_train",
                    source="/opt/ml/processing/output/gold/train",
                    destination=f"s3://{GOLD_BUCKET}/training-pairs/train/",
                ),
                ProcessingOutput(
                    output_name="gold_eval",
                    source="/opt/ml/processing/output/gold/eval",
                    destination=f"s3://{GOLD_BUCKET}/training-pairs/eval/",
                ),
            ],
        ),
        depends_on=[step_preprocessing],
    )

    ###########################################################################
    # Step 3: Training — QLoRA + RAFT (Spot instance)
    ###########################################################################

    # Retrieve the S3 URI for the JumpStart pre-trained model weights.
    # JumpStart only auto-delivers weights to SM_CHANNEL_MODEL when using its
    # built-in training scripts. With a custom entry_point (train.py), we must
    # pass the weights explicitly as a "model" training input channel.
    jumpstart_model_uri = model_uris.retrieve(
        model_id=JUMPSTART_MODEL_ID,
        model_version="*",
        model_scope="training",
        sagemaker_session=sm_session,
    )
    logger.info("JumpStart model URI: %s", jumpstart_model_uri)

    jumpstart_estimator = JumpStartEstimator(
        model_id=JUMPSTART_MODEL_ID,
        entry_point="train.py",
        source_dir=_make_training_source_dir(),
        role=ROLE_ARN,
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        # Managed Spot Training — up to 70% cost reduction.
        use_spot_instances=True,
        max_wait=86400,  # 24 hours max total wait including spot interruption.
        max_run=7200,    # 2 hours max actual training time.
        checkpoint_s3_uri=f"s3://{GOLD_BUCKET}/checkpoints/",
        hyperparameters={
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
        estimator=jumpstart_estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=step_synthesize.properties.ProcessingOutputConfig.Outputs["gold_train"].S3Output.S3Uri,
                content_type="application/x-ndjson",
            ),
            # Explicit model channel delivers JumpStart weights to SM_CHANNEL_MODEL
            # so train.py loads from AWS-managed S3 without HuggingFace Hub access.
            "model": sagemaker.inputs.TrainingInput(
                s3_data=jumpstart_model_uri,
                content_type="application/x-tar",
                s3_data_type="S3Prefix",
            ),
        },
        depends_on=[step_embed, step_synthesize],
    )

    ###########################################################################
    # Step 4: Evaluate — 3-suite evaluation
    ###########################################################################

    # GPU instance: evaluate.py loads the merged 3B bfloat16 model (~6 GB) and
    # runs inference across 3 evaluation suites. ml.g4dn.xlarge (16 GB VRAM)
    # handles the 3B model comfortably and is cheaper than ml.g5.2xlarge.
    eval_processor = PyTorchProcessor(
        framework_version="2.1.0",
        py_version="py310",
        role=ROLE_ARN,
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
        env=_CONTAINER_ENV,
    )

    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )

    step_evaluate = ProcessingStep(
        name="EvaluateRAFT",
        step_args=eval_processor.run(
            code="evaluate.py",
            source_dir=_make_processing_source_dir(STEPS_DIR / "evaluate.py"),
            inputs=[
                ProcessingInput(
                    source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                    destination="/opt/ml/processing/input/model",
                    input_name="model",
                ),
                ProcessingInput(
                    source=step_synthesize.properties.ProcessingOutputConfig.Outputs["gold_eval"].S3Output.S3Uri,
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
        ),
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
        estimator=jumpstart_estimator,
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
        parameters=[input_data_uri, model_approval_status],
        steps=[
            step_preprocessing,
            step_embed,
            step_synthesize,
            step_train,
            step_evaluate,
            step_condition,
        ],
        sagemaker_session=sm_session,
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

    # Clean up temp dirs created by _make_processing_source_dir() and
    # _make_training_source_dir() — upsert() has already uploaded them to S3.
    for tmp_path in Path(tempfile.gettempdir()).glob("chitrakatha_proc_*"):
        shutil.rmtree(tmp_path, ignore_errors=True)
    for tmp_path in Path(tempfile.gettempdir()).glob("chitrakatha_train_*"):
        shutil.rmtree(tmp_path, ignore_errors=True)

    if args.execute:
        execution = pipeline.start(
            parameters={"InputDataUri": args.input_uri}
        )
        logger.info("Pipeline execution started: %s", execution.arn)
        print(f"PIPELINE_EXECUTION_ARN={execution.arn}")


if __name__ == "__main__":
    main()
