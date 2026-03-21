"""SageMaker Lineage Tracking Utilities.

Why: While SageMaker Pipelines automatically tracks standard inputs/outputs
     (e.g., TrainingJob reading from an S3 channel), custom artifacts like
     our S3 Vectors Index or the dynamically generated RAFT pairs benefit
     from explicit lineage tracking. This ensures full auditability from
     the raw comic PDF down to the deployed Serverless Endpoint.

Constraints:
    - Fails gracefully if the SageMaker SDK or Lineage API is unavailable.
    - Uses 'AssociatedWith' and 'ContributedTo' edge types to link
      custom artifacts to SageMaker Pipeline pipeline executions.
"""

from __future__ import annotations

import logging
from typing import Any

import boto3

logger = logging.getLogger(__name__)


def track_artifact(
    artifact_uri: str,
    artifact_name: str,
    artifact_type: str,
    action_arn: str,
    direction: str = "Input",
    sagemaker_session: Any = None,
) -> None:
    """Link a custom S3 URI or external asset to a SageMaker Action.

    Args:
        artifact_uri: The S3 URI or string identifier of the asset
            (e.g., 's3://chitrakatha-vectors/index').
        artifact_name: Human-readable name (e.g., 'S3_Vector_Index').
        artifact_type: Type descriptor (e.g., 'DataSet', 'Model', 'Index').
        action_arn: The ARN of the SageMaker ProcessingJob, TrainingJob,
            or Pipeline Execution.
        direction: 'Input' (Produced the action) or 'Output' (Created by the action).
        sagemaker_session: Optional sagemaker Session. Defaults to boto3.
    """
    try:
        from sagemaker.lineage.action import Action  # noqa: PLC0415
        from sagemaker.lineage.artifact import Artifact  # noqa: PLC0415
        from sagemaker.lineage.association import Association  # noqa: PLC0415
        from sagemaker.session import Session  # noqa: PLC0415
    except ImportError:
        logger.warning(
            "sagemaker SDK not available — lineage not tracked for %s", artifact_name
        )
        return

    session = sagemaker_session or Session()

    try:
        # Create or retrieve the Artifact record in SageMaker Lineage
        artifact = Artifact.create(
            artifact_name=artifact_name,
            source_uri=artifact_uri,
            artifact_type=artifact_type,
            sagemaker_session=session,
        )

        # Retrieve the Action record (Jobs are automatically created as Actions)
        # We need the Action ARN formatted correctly for the association.
        # Often, action_arn is just the job ARN itself.

        # Create the Directed Edge
        if direction.lower() == "input":
            source_arn = artifact.artifact_arn
            dest_arn = action_arn
            assoc_type = "ContributedTo"
        else:
            source_arn = action_arn
            dest_arn = artifact.artifact_arn
            assoc_type = "Produced"

        Association.create(
            source_arn=source_arn,
            destination_arn=dest_arn,
            association_type=assoc_type,
            sagemaker_session=session,
        )
        
        logger.info(
            "Lineage tracked: %s '%s' %s %s",
            artifact_type,
            artifact_name,
            assoc_type,
            action_arn,
        )

    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not track lineage for artifact '%s': %s",
            artifact_name,
            exc,
            exc_info=True,
        )
