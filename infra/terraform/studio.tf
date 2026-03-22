###############################################################################
# Project Chitrakatha — SageMaker Studio Domain
#
# Why: Studio gives the team a browser-based control plane for:
#       - Inspecting pipeline executions and step logs
#       - Manually triggering pipeline runs (without GitHub Actions)
#       - Browsing the Model Registry and approving/rejecting packages
#       - Running ad-hoc notebooks against the same IAM role and S3 buckets
#         used by the pipeline — no local AWS config required
#
# Auth mode: IAM — users sign in with their existing AWS credentials via the
#   console. No SSO/Identity Center setup required.
#
# Network: PublicInternetOnly — Studio Jupyter server connects directly to
#   the internet. Processing/Training jobs still run in SageMaker-managed
#   networking and access S3/Bedrock over AWS backbone. VpcOnly would require
#   NAT gateways and VPC endpoints, adding ~$50/month with no security benefit
#   for a dev project.
#
# Execution role: reuses the existing sagemaker_execution role so notebooks
#   and manually triggered jobs have the same permissions as CI/CD jobs.
###############################################################################

resource "aws_sagemaker_domain" "chitrakatha" {
  domain_name             = "${var.project_name}-studio"
  auth_mode               = "IAM"
  vpc_id                  = aws_vpc.chitrakatha.id
  subnet_ids              = [aws_subnet.public_a.id, aws_subnet.public_b.id]
  app_network_access_type = "PublicInternetOnly"

  default_user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn

    # Shut down JupyterLab after 60 minutes of inactivity (minimum allowed).
    # This prevents charges (~$0.005/hr for JupyterServer) when Studio is left
    # open in the browser without active use.
    jupyter_lab_app_settings {
      app_lifecycle_management {
        idle_settings {
          idle_timeout_in_minutes     = 60
          min_idle_timeout_in_minutes = 60
          max_idle_timeout_in_minutes = 120
        }
      }
    }
  }

  # Retain user data (notebooks, experiments) even after terraform destroy.
  retention_policy {
    home_efs_file_system = "Retain"
  }

  tags = {
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_sagemaker_user_profile" "default" {
  domain_id         = aws_sagemaker_domain.chitrakatha.id
  user_profile_name = var.studio_user_profile_name

  user_settings {
    execution_role = aws_iam_role.sagemaker_execution.arn
  }

  tags = {
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}
