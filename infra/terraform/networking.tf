###############################################################################
# Project Chitrakatha — Minimal VPC for SageMaker Studio
#
# Why: SageMaker Studio domains require a VPC + subnets even when
#      app_network_access_type = "PublicInternetOnly" (Studio's JupyterServer
#      connects directly to the internet via SageMaker's managed network, but
#      the domain API still requires these parameters).
#
#      Two public subnets across two AZs provide redundancy for any
#      processing/training jobs launched from Studio notebooks.
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
