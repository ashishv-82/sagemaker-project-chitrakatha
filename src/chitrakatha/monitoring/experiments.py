"""SageMaker Experiments logging wrapper.

Why: SageMaker Experiments provides automatic lineage from data to model —
     every training run's hyperparameters and evaluation metrics are stored
     against the Experiment, enabling comparison across pipeline runs.
     Wrapping the SDK call here means step scripts never import
     ``sagemaker.experiments`` directly, making them easier to test (mock
     this module rather than the SageMaker SDK).

     If the Experiments SDK is not available (e.g. thin Processing containers
     that don't have the full SageMaker SDK), all calls degrade gracefully
     to a log statement — the pipeline never fails due to a metrics-logging
     issue.

Constraints:
    - Never raises — any failure is caught and logged as WARNING.
    - Works outside SageMaker (e.g. local dev) by logging to stdout.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def log_metrics(run_name: str, metrics: dict[str, float]) -> None:
    """Log numeric metrics to the active SageMaker Experiments run.

    Args:
        run_name: The SageMaker Experiments run name (from the
            ``SAGEMAKER_EXPERIMENT_RUN`` env var injected by the pipeline).
        metrics: Dict of metric name → numeric value. All values are cast
            to float before logging.
    """
    if not run_name:
        logger.debug("No experiment run name — logging metrics to stdout only: %s", metrics)
        return

    try:
        from sagemaker.experiments.run import Run  # noqa: PLC0415

        with Run(run_name=run_name, sagemaker_session=None) as run:
            for name, value in metrics.items():
                run.log_metric(name=name, value=float(value))
        logger.info("Logged metrics to experiment run '%s': %s", run_name, metrics)

    except ImportError:
        # SageMaker SDK not available — log metrics locally only.
        logger.warning(
            "sagemaker SDK not available — metrics not persisted to Experiments. "
            "Metrics: %s",
            metrics,
        )
    except Exception as exc:  # noqa: BLE001
        # Never fail the pipeline step because of an Experiments logging error.
        logger.warning(
            "Could not log metrics to Experiments run '%s': %s. Metrics: %s",
            run_name, exc, metrics,
        )


def log_hyperparameters(run_name: str, hyperparameters: dict[str, object]) -> None:
    """Log training hyperparameters to the SageMaker Experiments run.

    Args:
        run_name: Experiments run name.
        hyperparameters: Dict of hyperparameter name → value (any type;
            values are cast to str for Experiments compatibility).
    """
    if not run_name:
        logger.debug("No experiment run name — logging hyperparameters locally: %s", hyperparameters)
        return

    try:
        from sagemaker.experiments.run import Run  # noqa: PLC0415

        with Run(run_name=run_name, sagemaker_session=None) as run:
            for name, value in hyperparameters.items():
                run.log_parameter(name=name, value=str(value))
        logger.info(
            "Logged hyperparameters to experiment run '%s': %s", run_name, hyperparameters
        )

    except ImportError:
        logger.warning(
            "sagemaker SDK not available — hyperparameters not persisted. "
            "Hyperparameters: %s",
            hyperparameters,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Could not log hyperparameters to Experiments run '%s': %s",
            run_name, exc,
        )
