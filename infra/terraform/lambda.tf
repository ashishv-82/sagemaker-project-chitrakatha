###############################################################################
# Project Chitrakatha — Lambda Bridge & API Gateway
#
# Why: Connects public internet -> HTTP API Gateway -> Lambda -> SageMaker
#      Serverless Endpoint. Provides a fully serverless, scale-to-zero Web API
#      for the RAG application without persistent load balancers.
#
# Constraints:
#   - Pydantic v2 dependency must be packaged with the ZIP. Here we define
#     a null_resource to run pip install locally before zipping.
#   - API Gateway uses HTTP APIs (v2) which are 71% cheaper than REST APIs.
###############################################################################

# 1. Package dependencies locally before zipping source code.
resource "null_resource" "pip_install_lambda" {
  triggers = {
    requirements = filemd5("${path.module}/../../serving/lambda/requirements.txt")
    handler      = filemd5("${path.module}/../../serving/lambda/handler.py")
  }

  provisioner "local-exec" {
    command = <<EOT
      mkdir -p ${path.module}/../../serving/lambda/package
      pip3 install -r ${path.module}/../../serving/lambda/requirements.txt -t ${path.module}/../../serving/lambda/package --upgrade
      cp ${path.module}/../../serving/lambda/handler.py ${path.module}/../../serving/lambda/package/
    EOT
  }
}

# 2. Create the deployment ZIP archive.
data "archive_file" "lambda_zip" {
  depends_on  = [null_resource.pip_install_lambda]
  type        = "zip"
  source_dir  = "${path.module}/../../serving/lambda/package"
  output_path = "${path.module}/../../serving/lambda/function.zip"
}

# 3. Provision the Lambda function.
resource "aws_lambda_function" "api_bridge" {
  function_name    = "${var.project_name}-rag-bridge"
  role             = aws_iam_role.lambda_execution.arn
  handler          = "handler.handler"
  runtime          = "python3.12"
  filename         = data.archive_file.lambda_zip.output_path
  source_code_hash = data.archive_file.lambda_zip.output_base64sha256
  timeout     = 60
  memory_size = 256

  vpc_config {
    subnet_ids         = [aws_subnet.private_a.id, aws_subnet.private_b.id]
    security_group_ids = [aws_security_group.lambda.id]
  }

  environment {
    variables = {
      DB_SECRET_ARN          = aws_secretsmanager_secret.rds_credentials.arn
      BEDROCK_QWEN3_MODEL_ID = "qwen.qwen3-next-80b-a3b"
      BEDROCK_EMBED_MODEL_ID = "amazon.titan-embed-text-v2:0"
    }
  }

  tags = {
    Purpose = "Public RAG API Bridge"
  }
}

###############################################################################
# API Gateway HTTP API v2
###############################################################################

resource "aws_apigatewayv2_api" "rag_api" {
  name          = "${var.project_name}-http-api"
  protocol_type = "HTTP"

  cors_configuration {
    allow_origins = ["*"] # Adjust in production
    allow_methods = ["POST", "OPTIONS"]
    allow_headers = ["content-type", "x-amz-date", "authorization", "x-amz-security-token"]
    max_age       = 300
  }
}

resource "aws_apigatewayv2_stage" "default" {
  api_id      = aws_apigatewayv2_api.rag_api.id
  name        = "$default"
  auto_deploy = true

  access_log_settings {
    destination_arn = aws_cloudwatch_log_group.api_gateway.arn
    format = jsonencode({
      requestId      = "$context.requestId"
      ip             = "$context.identity.sourceIp"
      requestTime    = "$context.requestTime"
      httpMethod     = "$context.httpMethod"
      routeKey       = "$context.routeKey"
      status         = "$context.status"
      protocol       = "$context.protocol"
      responseLength = "$context.responseLength"
    })
  }
}

resource "aws_apigatewayv2_integration" "lambda_integration" {
  api_id             = aws_apigatewayv2_api.rag_api.id
  integration_uri    = aws_lambda_function.api_bridge.invoke_arn
  integration_type   = "AWS_PROXY"
  integration_method = "POST"
}

resource "aws_apigatewayv2_route" "post_query" {
  api_id    = aws_apigatewayv2_api.rag_api.id
  route_key = "POST /v1/query"
  target    = "integrations/${aws_apigatewayv2_integration.lambda_integration.id}"
}

# Allow API Gateway to invoke the Lambda function
resource "aws_lambda_permission" "api_gateway_invoke" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api_bridge.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.rag_api.execution_arn}/*/*"
}

# CloudWatch Log Group for API Gateway Access Logs 
resource "aws_cloudwatch_log_group" "api_gateway" {
  name              = "/aws/apigateway/${var.project_name}-http-api"
  retention_in_days = 30
}
