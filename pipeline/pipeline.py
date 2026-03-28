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
    Qwen/Qwen2.5-3B-Instruct (HuggingFace Hub, Apache 2.0 — no token required)
    JumpStart does not support Llama 3.2 fine-tuning in ap-southeast-2.
    Qwen2.5 has stronger multilingual (Hindi) capability than Llama 3.2 at 3B scale.

Constraints:
    - All ARNs and bucket names from environment (Terraform outputs) — no hardcoding.
    - Tags: Project: Chitrakatha, CostCenter: MLOps-Research on every resource.
    - On-demand training (spot quota is 0 in ap-southeast-2 by default).
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import tempfile
from pathlib import Path

import sagemaker
from sagemaker.model import Model
from sagemaker.pytorch import PyTorch
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
DB_SECRET_ARN = os.environ["DB_SECRET_ARN"]

PIPELINE_NAME = "chitrakatha-mlops-pipeline"

# Source directory for pipeline step scripts (relative to this file).
STEPS_DIR = Path(__file__).parent / "steps"
ROOT_DIR = Path(__file__).parent.parent

# S3 prefix within SILVER_BUCKET where the chitrakatha package is published so
# processing containers can import it via PYTHONPATH (source_dir is unsupported
# by ScriptProcessor.run(); FrameworkProcessor subclasses also fail to propagate
# it through the @runnable_by_pipeline replay in newer SDK versions).
PIPELINE_ASSETS_PREFIX = "pipeline-assets"

# Resource tags applied to every SageMaker resource created by this pipeline.
RESOURCE_TAGS = [
    {"Key": "Project", "Value": "Chitrakatha"},
    {"Key": "CostCenter", "Value": "MLOps-Research"},
]


def _sync_chitrakatha_to_s3() -> str:
    """Upload the chitrakatha source package to S3 for use in processing containers.

    ``source_dir`` in ``FrameworkProcessor.run()`` / ``ScriptProcessor.run()``
    is not reliably supported across SageMaker SDK versions in the
    ``@runnable_by_pipeline`` replay phase.  Instead we upload the package
    once per pipeline run and mount it as a ``ProcessingInput`` at
    ``/opt/ml/processing/input/src``, then set
    ``PYTHONPATH=/opt/ml/processing/input/src`` so ``import chitrakatha`` works.

    Returns:
        S3 URI of the parent src/ directory
        (e.g. ``s3://<bucket>/pipeline-assets/src/``).
    """
    import boto3 as _boto3

    s3_client = _boto3.client("s3", region_name=AWS_REGION)
    src_dir = ROOT_DIR / "src"
    chitrakatha_dir = src_dir / "chitrakatha"
    s3_prefix = f"{PIPELINE_ASSETS_PREFIX}/src"

    for file_path in chitrakatha_dir.rglob("*"):
        if file_path.is_file() and "__pycache__" not in str(file_path):
            relative = file_path.relative_to(src_dir)
            key = f"{s3_prefix}/{relative}"
            s3_client.upload_file(str(file_path), SILVER_BUCKET, key)

    s3_uri = f"s3://{SILVER_BUCKET}/{s3_prefix}/"
    logger.info("Synced chitrakatha package to %s", s3_uri)
    return s3_uri


def _make_training_source_dir() -> str:
    """Bundle train.py, requirements.txt, and the chitrakatha package.

    The PyTorchEstimator installs packages from requirements.txt automatically
    if it is present in source_dir. requirements.txt lives in pipeline/ (not
    pipeline/steps/), so it must be explicitly copied into the temp dir.
    """
    tmp = Path(tempfile.mkdtemp(prefix="chitrakatha_train_"))
    for item in STEPS_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, tmp / item.name)
    # Copy pipeline/requirements.txt so SageMaker installs training dependencies.
    reqs = Path(__file__).parent / "requirements.txt"
    if reqs.exists():
        shutil.copy2(reqs, tmp / "requirements.txt")
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
    # PYTHONPATH points to the mounted chitrakatha package (uploaded to S3 by
    # _sync_chitrakatha_to_s3() in main() and mounted via ProcessingInput).
    _CONTAINER_ENV: dict[str, str] = {
        "S3_BRONZE_BUCKET": BRONZE_BUCKET,
        "S3_SILVER_BUCKET": SILVER_BUCKET,
        "S3_GOLD_BUCKET": GOLD_BUCKET,
        "DB_SECRET_ARN": DB_SECRET_ARN,
        "KMS_KEY_ARN": KMS_KEY_ARN,
        "SAGEMAKER_ROLE_ARN": ROLE_ARN,
        "AWS_REGION": AWS_REGION,
        "PYTHONPATH": "/opt/ml/processing/input/src",
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

    # chitrakatha package is uploaded to S3 by _sync_chitrakatha_to_s3() before
    # create_pipeline() is called. Every processing step mounts it at
    # /opt/ml/processing/input/src so that `import chitrakatha` resolves via
    # PYTHONPATH (set in _CONTAINER_ENV above). This avoids source_dir which is
    # not reliably supported by ScriptProcessor/FrameworkProcessor.run() during
    # the @runnable_by_pipeline replay phase in SageMaker SDK >= 2.200.
    chitrakatha_input = ProcessingInput(
        source=f"s3://{SILVER_BUCKET}/{PIPELINE_ASSETS_PREFIX}/src/",
        destination="/opt/ml/processing/input/src",
        input_name="chitrakatha-src",
    )

    step_preprocessing = ProcessingStep(
        name="Preprocessing",
        step_args=sklearn_processor.run(
            code=str(STEPS_DIR / "preprocessing.py"),
            inputs=[
                ProcessingInput(
                    source=input_data_uri,
                    destination="/opt/ml/processing/input/bronze",
                    input_name="bronze",
                ),
                chitrakatha_input,
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
    # ScriptProcessor is used (not FrameworkProcessor subclass) because source_dir
    # is avoided — chitrakatha is mounted via ProcessingInput + PYTHONPATH instead.
    embed_processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-cpu-py310-ubuntu20.04-sagemaker",
        instance_type="ml.m5.2xlarge",
        instance_count=1,
        command=["python3"],
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
        env={**_CONTAINER_ENV, "SAGEMAKER_EXPERIMENT_RUN": "chitrakatha-pipeline-run"},
    )

    step_embed = ProcessingStep(
        name="EmbedAndIndex",
        step_args=embed_processor.run(
            code=str(STEPS_DIR / "embed_and_index.py"),
            inputs=[
                ProcessingInput(
                    source=step_preprocessing.properties.ProcessingOutputConfig.Outputs["corpus"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/corpus",
                    input_name="corpus",
                ),
                chitrakatha_input,
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
            code=str(STEPS_DIR / "synthesize_pairs.py"),
            inputs=[
                ProcessingInput(
                    source=step_preprocessing.properties.ProcessingOutputConfig.Outputs["training"].S3Output.S3Uri,
                    destination="/opt/ml/processing/input/training",
                    input_name="training",
                ),
                chitrakatha_input,
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
    # Step 3: Training — QLoRA + RAFT
    # Uses PyTorchEstimator (HuggingFace Hub) instead of JumpStartEstimator
    # because JumpStart does not support Llama 3.2 fine-tuning in ap-southeast-2.
    # The HF token is read from Secrets Manager inside train.py at runtime.
    ###########################################################################

    pytorch_estimator = PyTorch(
        entry_point="train.py",
        source_dir=_make_training_source_dir(),
        role=ROLE_ARN,
        framework_version="2.1.0",
        py_version="py310",
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        use_spot_instances=False,
        max_run=7200,
        checkpoint_s3_uri=f"s3://{GOLD_BUCKET}/checkpoints/",
        hyperparameters={
            "EPOCHS": "3",
            "BATCH_SIZE": "4",
            "LEARNING_RATE": "2e-4",
            "LORA_R": "16",
            "LORA_ALPHA": "32",
        },
        environment={
            "SAGEMAKER_EXPERIMENT_RUN": "chitrakatha-pipeline-run",
        },
        output_path=f"s3://{GOLD_BUCKET}/model-artifacts/",
        sagemaker_session=sm_session,
        tags=RESOURCE_TAGS,
    )

    step_train = TrainingStep(
        name="TrainQLoRARAFT",
        estimator=pytorch_estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=step_synthesize.properties.ProcessingOutputConfig.Outputs["gold_train"].S3Output.S3Uri,
                content_type="application/x-ndjson",
            ),
            # Pre-cached Qwen2.5-3B weights — avoids HuggingFace download at training time.
            # Mounted at /opt/ml/input/data/model/ → SM_CHANNEL_MODEL env var.
            "model": sagemaker.inputs.TrainingInput(
                s3_data=f"s3://{GOLD_BUCKET}/base-models/qwen2.5-3b-instruct/",
                input_mode="File",
            ),
        },
        depends_on=[step_synthesize],
    )

    ###########################################################################
    # Step 4: Evaluate — 3-suite evaluation
    ###########################################################################

    # GPU instance: evaluate.py loads the merged 3B bfloat16 model (~6 GB) and
    # runs inference across 3 evaluation suites. ml.g4dn.xlarge (16 GB VRAM)
    # handles the 3B model comfortably and is cheaper than ml.g5.2xlarge.
    eval_processor = ScriptProcessor(
        role=ROLE_ARN,
        image_uri=f"763104351884.dkr.ecr.{AWS_REGION}.amazonaws.com/pytorch-training:2.1.0-gpu-py310-cu121-ubuntu20.04-sagemaker",
        instance_type="ml.g4dn.xlarge",
        instance_count=1,
        command=["python3"],
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
            code=str(STEPS_DIR / "evaluate.py"),
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
                chitrakatha_input,
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
        estimator=pytorch_estimator,
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

    # Upload the chitrakatha package to S3 before building the pipeline so that
    # every ProcessingStep can mount it via ProcessingInput + PYTHONPATH.
    _sync_chitrakatha_to_s3()

    pipeline = create_pipeline()

    pipeline.upsert(role_arn=ROLE_ARN, tags=RESOURCE_TAGS)
    logger.info("Pipeline '%s' upserted successfully.", PIPELINE_NAME)

    # Clean up temp dirs created by _make_training_source_dir().
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
