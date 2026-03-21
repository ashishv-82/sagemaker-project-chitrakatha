###############################################################################
# Project Chitrakatha — CloudWatch Alarms & Dashboard
#
# Why: SageMaker Serverless Inference endpoints have cold-start latency and
#      can fail silently under error conditions. CloudWatch alarms provide
#      real-time alerts before costs spiral or users experience degraded service.
#
# Alarms:
#   1. ModelInvocationErrors  — 4xx/5xx errors > 5 in a 5-min window
#   2. Cold-start latency P99 — > 30s (serverless cold-start threshold)
#   3. Error rate             — > 1% of invocations are errors
#
# Constraints:
#   - Alarms use `treat_missing_data = "notBreaching"` because endpoints
#     scale to zero (no invocations = no error, not an alarm condition).
#   - SNS topic ARN is optional (empty in dev); alarm still exists but
#     has no notification action.
###############################################################################

locals {
  # Endpoint name must match what deploy_endpoint.py creates (Phase 4).
  endpoint_name = "${var.project_name}-rag-serverless"

  # Alarm actions only fire if an SNS topic ARN is provided.
  alarm_actions = var.alarm_sns_topic_arn != "" ? [var.alarm_sns_topic_arn] : []
}

###############################################################################
# Alarm 1: Model Invocation Errors (absolute count)
###############################################################################

resource "aws_cloudwatch_metric_alarm" "endpoint_invocation_errors" {
  alarm_name          = "${var.project_name}-endpoint-invocation-errors"
  alarm_description   = "Fires when SageMaker serverless endpoint returns > 5 errors in a 5-minute window. Indicates model server crash or invalid input format."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 1
  metric_name         = "ModelInvocationErrors"
  namespace           = "AWS/SageMaker"
  period              = 300 # 5 minutes
  statistic           = "Sum"
  threshold           = 5

  dimensions = {
    EndpointName = local.endpoint_name
    VariantName  = "AllTraffic"
  }

  # Scale-to-zero: no data = no invocations = not an alarm condition.
  treat_missing_data = "notBreaching"

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}

###############################################################################
# Alarm 2: Cold-Start Latency P99 > 30s
###############################################################################

resource "aws_cloudwatch_metric_alarm" "endpoint_cold_start_latency" {
  alarm_name          = "${var.project_name}-endpoint-cold-start-p99"
  alarm_description   = "Fires when P99 cold-start (ModelSetupTime) exceeds 30 seconds. Indicates the serverless container is too large or model loading is unoptimised."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  metric_name         = "ModelSetupTime"
  namespace           = "AWS/SageMaker"
  period              = 300
  extended_statistic  = "p99"
  threshold           = 30000 # milliseconds

  dimensions = {
    EndpointName = local.endpoint_name
    VariantName  = "AllTraffic"
  }

  treat_missing_data = "notBreaching"

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}

###############################################################################
# Alarm 3: Error Rate > 1% (ratio of errors to total invocations)
###############################################################################

resource "aws_cloudwatch_metric_alarm" "endpoint_error_rate" {
  alarm_name          = "${var.project_name}-endpoint-error-rate"
  alarm_description   = "Fires when the ratio of ModelInvocationErrors to Invocations exceeds 1% over 10 minutes. Indicates a systematic model or RAG retrieval issue."
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = 2
  threshold           = 1 # percent

  # Use a metric math expression to compute the error rate percentage.
  metric_query {
    id          = "error_rate"
    expression  = "(errors / MAX([errors, invocations])) * 100"
    label       = "Error Rate (%)"
    return_data = true
  }

  metric_query {
    id = "errors"
    metric {
      metric_name = "ModelInvocationErrors"
      namespace   = "AWS/SageMaker"
      period      = 300
      stat        = "Sum"
      dimensions = {
        EndpointName = local.endpoint_name
        VariantName  = "AllTraffic"
      }
    }
  }

  metric_query {
    id = "invocations"
    metric {
      metric_name = "Invocations"
      namespace   = "AWS/SageMaker"
      period      = 300
      stat        = "Sum"
      dimensions = {
        EndpointName = local.endpoint_name
        VariantName  = "AllTraffic"
      }
    }
  }

  treat_missing_data = "notBreaching"

  alarm_actions = local.alarm_actions
  ok_actions    = local.alarm_actions
}

###############################################################################
# CloudWatch Dashboard — ChitrakathaMLOpsDashboard
###############################################################################

resource "aws_cloudwatch_dashboard" "chitrakatha" {
  dashboard_name = "ChitrakathaMLOpsDashboard"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Endpoint Invocations"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", local.endpoint_name, "VariantName", "AllTraffic"]
          ]
          period = 300
          stat   = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Error Rate"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            ["AWS/SageMaker", "ModelInvocationErrors", "EndpointName", local.endpoint_name, "VariantName", "AllTraffic"]
          ]
          period = 300
          stat   = "Sum"
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title  = "Cold-Start Latency P99 (ms)"
          view   = "timeSeries"
          region = var.aws_region
          metrics = [
            [{ "expression" = "METRICS()", "label" = "ModelSetupTime", "id" = "e1" }],
            ["AWS/SageMaker", "ModelSetupTime", "EndpointName", local.endpoint_name, "VariantName", "AllTraffic", { "id" = "m1", "stat" = "p99", "visible" = false }]
          ]
          period = 300
        }
      },
      {
        type   = "alarm"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title = "Alarm Status"
          alarms = [
            aws_cloudwatch_metric_alarm.endpoint_invocation_errors.arn,
            aws_cloudwatch_metric_alarm.endpoint_cold_start_latency.arn,
            aws_cloudwatch_metric_alarm.endpoint_error_rate.arn,
          ]
        }
      }
    ]
  })
}
