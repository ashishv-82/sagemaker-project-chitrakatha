###############################################################################
# Project Chitrakatha — AWS Secrets Manager
#
# Why: The synthetic data API key (used by Claude during RAFT pair synthesis)
#      must never be hardcoded in Python or Terraform. Secrets Manager provides
#      automatic rotation, fine-grained IAM access, and CloudTrail audit trails.
#
# Workflow:
#   1. Terraform creates the secret with a placeholder value ("REPLACE_ME").
#   2. CI/CD injects the real value via `aws secretsmanager put-secret-value`
#      using OIDC-federated credentials — never stored in git or env files.
#   3. Python code retrieves the value at runtime via boto3 — never in config.
#
# Constraints:
#   - Secret name follows `{project}/` prefix scoped in the IAM policy.
#   - Recovery window: 30 days (allows accidental deletion recovery).
#   - KMS encryption: uses the project CMK, not the default Secrets Manager key.
###############################################################################

resource "aws_secretsmanager_secret" "synthetic_data_api_key" {
  name        = "${var.project_name}/synthetic_data_api_key"
  description = "API key used by synthesize_training_pairs.py when calling Bedrock Claude for RAFT data generation. Real value injected by CI/CD post-deployment."

  # Encrypt with the project CMK — consistent with all other data assets.
  kms_key_id = aws_kms_key.chitrakatha.arn

  # 30-day recovery window prevents irreversible accidental deletion.
  recovery_window_in_days = 30
}

resource "aws_secretsmanager_secret_version" "synthetic_data_api_key_placeholder" {
  secret_id = aws_secretsmanager_secret.synthetic_data_api_key.id

  # Placeholder value — CI/CD overwrites this with the real key after `apply`.
  # This block is `ignore_changes`-guarded so Terraform never reverts a real key.
  secret_string = jsonencode({
    api_key = "REPLACE_ME_VIA_CICD"
    note    = "Inject real value using: aws secretsmanager put-secret-value --secret-id ${var.project_name}/synthetic_data_api_key --secret-string '{\"api_key\":\"<real_key>\"}'"
  })

  lifecycle {
    # Prevent Terraform from reverting the secret back to the placeholder
    # after CI/CD has injected the real value.
    ignore_changes = [secret_string]
  }
}
