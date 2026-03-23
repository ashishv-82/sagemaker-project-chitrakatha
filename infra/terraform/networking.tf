###############################################################################
# Project Chitrakatha — VPC, Subnets, Security Groups, VPC Endpoints
#
# Public subnets (10.0.1/2.0/24): SageMaker Studio + training jobs.
# Private subnets (10.0.3/4.0/24): RDS PostgreSQL + Lambda bridge.
#
# Option B networking: RDS never reachable from internet; Lambda reaches
# Bedrock and Secrets Manager via VPC Interface Endpoints (no NAT Gateway).
# S3 uses a Gateway Endpoint (free). ~$16 AUD/month for interface endpoints.
###############################################################################

resource "aws_vpc" "chitrakatha" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = {
    Name       = "${var.project_name}-vpc"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_internet_gateway" "chitrakatha" {
  vpc_id = aws_vpc.chitrakatha.id

  tags = {
    Name       = "${var.project_name}-igw"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_subnet" "public_a" {
  vpc_id                  = aws_vpc.chitrakatha.id
  cidr_block              = "10.0.1.0/24"
  availability_zone       = "${var.aws_region}a"
  map_public_ip_on_launch = true

  tags = {
    Name       = "${var.project_name}-public-a"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_subnet" "public_b" {
  vpc_id                  = aws_vpc.chitrakatha.id
  cidr_block              = "10.0.2.0/24"
  availability_zone       = "${var.aws_region}b"
  map_public_ip_on_launch = true

  tags = {
    Name       = "${var.project_name}-public-b"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.chitrakatha.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.chitrakatha.id
  }

  tags = {
    Name       = "${var.project_name}-public-rt"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_route_table_association" "public_a" {
  subnet_id      = aws_subnet.public_a.id
  route_table_id = aws_route_table.public.id
}

resource "aws_route_table_association" "public_b" {
  subnet_id      = aws_subnet.public_b.id
  route_table_id = aws_route_table.public.id
}

###############################################################################
# Private Subnets — RDS + Lambda (no route to internet)
###############################################################################

resource "aws_subnet" "private_a" {
  vpc_id            = aws_vpc.chitrakatha.id
  cidr_block        = "10.0.3.0/24"
  availability_zone = "${var.aws_region}a"

  tags = {
    Name       = "${var.project_name}-private-a"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_subnet" "private_b" {
  vpc_id            = aws_vpc.chitrakatha.id
  cidr_block        = "10.0.4.0/24"
  availability_zone = "${var.aws_region}b"

  tags = {
    Name       = "${var.project_name}-private-b"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_route_table" "private" {
  vpc_id = aws_vpc.chitrakatha.id

  tags = {
    Name       = "${var.project_name}-private-rt"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_route_table_association" "private_a" {
  subnet_id      = aws_subnet.private_a.id
  route_table_id = aws_route_table.private.id
}

resource "aws_route_table_association" "private_b" {
  subnet_id      = aws_subnet.private_b.id
  route_table_id = aws_route_table.private.id
}

###############################################################################
# Security Groups
###############################################################################

# Security groups are created first with no inline rules.
# Rules are added separately via aws_security_group_rule to avoid circular references.

resource "aws_security_group" "lambda" {
  name        = "${var.project_name}-lambda-sg"
  description = "Lambda bridge: outbound to RDS and VPC endpoints only"
  vpc_id      = aws_vpc.chitrakatha.id

  tags = {
    Name       = "${var.project_name}-lambda-sg"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_security_group" "rds" {
  name        = "${var.project_name}-rds-sg"
  description = "RDS PostgreSQL: inbound from Lambda only"
  vpc_id      = aws_vpc.chitrakatha.id

  tags = {
    Name       = "${var.project_name}-rds-sg"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

resource "aws_security_group" "vpc_endpoints" {
  name        = "${var.project_name}-vpce-sg"
  description = "VPC endpoints: inbound HTTPS from Lambda only"
  vpc_id      = aws_vpc.chitrakatha.id

  tags = {
    Name       = "${var.project_name}-vpce-sg"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

# Lambda egress rules
resource "aws_security_group_rule" "lambda_to_rds" {
  type                     = "egress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.lambda.id
  source_security_group_id = aws_security_group.rds.id
  description              = "PostgreSQL to RDS"
}

resource "aws_security_group_rule" "lambda_to_vpce" {
  type                     = "egress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.lambda.id
  source_security_group_id = aws_security_group.vpc_endpoints.id
  description              = "HTTPS to VPC endpoints (Bedrock, Secrets Manager)"
}

# RDS ingress rule
resource "aws_security_group_rule" "rds_from_lambda" {
  type                     = "ingress"
  from_port                = 5432
  to_port                  = 5432
  protocol                 = "tcp"
  security_group_id        = aws_security_group.rds.id
  source_security_group_id = aws_security_group.lambda.id
  description              = "PostgreSQL from Lambda"
}

# VPC endpoints ingress rule
resource "aws_security_group_rule" "vpce_from_lambda" {
  type                     = "ingress"
  from_port                = 443
  to_port                  = 443
  protocol                 = "tcp"
  security_group_id        = aws_security_group.vpc_endpoints.id
  source_security_group_id = aws_security_group.lambda.id
  description              = "HTTPS from Lambda"
}

###############################################################################
# VPC Endpoints — Option B (no NAT Gateway)
###############################################################################

# Bedrock Runtime — interface endpoint (Lambda calls Bedrock via private network).
resource "aws_vpc_endpoint" "bedrock_runtime" {
  vpc_id              = aws_vpc.chitrakatha.id
  service_name        = "com.amazonaws.${var.aws_region}.bedrock-runtime"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = {
    Name       = "${var.project_name}-vpce-bedrock"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

# Secrets Manager — interface endpoint (Lambda fetches RDS credentials privately).
resource "aws_vpc_endpoint" "secretsmanager" {
  vpc_id              = aws_vpc.chitrakatha.id
  service_name        = "com.amazonaws.${var.aws_region}.secretsmanager"
  vpc_endpoint_type   = "Interface"
  subnet_ids          = [aws_subnet.private_a.id, aws_subnet.private_b.id]
  security_group_ids  = [aws_security_group.vpc_endpoints.id]
  private_dns_enabled = true

  tags = {
    Name       = "${var.project_name}-vpce-secretsmanager"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}

# S3 — gateway endpoint (free; allows Lambda to reach S3 without NAT).
resource "aws_vpc_endpoint" "s3" {
  vpc_id            = aws_vpc.chitrakatha.id
  service_name      = "com.amazonaws.${var.aws_region}.s3"
  vpc_endpoint_type = "Gateway"
  route_table_ids   = [aws_route_table.private.id]

  tags = {
    Name       = "${var.project_name}-vpce-s3"
    Project    = "Chitrakatha"
    CostCenter = "MLOps-Research"
  }
}
