# A2Z TSN/FRER Network Infrastructure - Terraform Configuration
# Production-ready cloud deployment for AWS EKS

terraform {
  required_version = ">= 1.5.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "s3" {
    bucket         = "a2z-terraform-state"
    key            = "production/terraform.tfstate"
    region         = "ap-northeast-2"  # Seoul region
    dynamodb_table = "a2z-terraform-locks"
    encrypt        = true
  }
}

# Provider configuration
provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Project     = "A2Z-TSN"
      Environment = var.environment
      ManagedBy   = "Terraform"
      CostCenter  = "NetworkOps"
    }
  }
}

provider "kubernetes" {
  host                   = module.eks.cluster_endpoint
  cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
  
  exec {
    api_version = "client.authentication.k8s.io/v1beta1"
    command     = "aws"
    args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
  }
}

provider "helm" {
  kubernetes {
    host                   = module.eks.cluster_endpoint
    cluster_ca_certificate = base64decode(module.eks.cluster_certificate_authority_data)
    
    exec {
      api_version = "client.authentication.k8s.io/v1beta1"
      command     = "aws"
      args = ["eks", "get-token", "--cluster-name", module.eks.cluster_name]
    }
  }
}

# Data sources
data "aws_availability_zones" "available" {
  state = "available"
}

data "aws_caller_identity" "current" {}

# Local variables
locals {
  cluster_name = "a2z-tsn-${var.environment}"
  
  common_tags = {
    ClusterName = local.cluster_name
    GitRepo     = "https://github.com/hwkim3330/a2z"
  }
  
  vpc_cidr = "10.0.0.0/16"
  azs      = slice(data.aws_availability_zones.available.names, 0, 3)
}

# VPC Module
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "${local.cluster_name}-vpc"
  cidr = local.vpc_cidr

  azs             = local.azs
  private_subnets = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 4, k)]
  public_subnets  = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 48)]
  intra_subnets   = [for k, v in local.azs : cidrsubnet(local.vpc_cidr, 8, k + 52)]

  enable_nat_gateway   = true
  single_nat_gateway   = false  # High availability
  enable_dns_hostnames = true
  enable_dns_support   = true

  # Enable VPC flow logs
  enable_flow_log                      = true
  create_flow_log_cloudwatch_log_group = true
  create_flow_log_cloudwatch_iam_role  = true
  flow_log_max_aggregation_interval    = 60

  # Kubernetes specific tags
  public_subnet_tags = {
    "kubernetes.io/role/elb"                    = 1
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb"           = 1
    "kubernetes.io/cluster/${local.cluster_name}" = "shared"
  }

  tags = local.common_tags
}

# EKS Module
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = local.cluster_name
  cluster_version = var.kubernetes_version

  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Add-ons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent    = true
      before_compute = true
      configuration_values = jsonencode({
        env = {
          ENABLE_PREFIX_DELEGATION = "true"
          WARM_PREFIX_TARGET       = "1"
        }
      })
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.intra_subnets

  # Security groups
  create_cluster_security_group = true
  create_node_security_group    = true

  # Node groups
  eks_managed_node_group_defaults = {
    ami_type       = "AL2_x86_64"
    instance_types = ["m5.large"]
    
    # Security
    attach_cluster_primary_security_group = true
    vpc_security_group_ids                = [aws_security_group.node_group_additional.id]
    
    # IAM
    iam_role_additional_policies = {
      AmazonSSMManagedInstanceCore = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
    }
  }

  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name = "general-nodes"
      
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      instance_types = ["m5.xlarge"]
      capacity_type  = "SPOT"
      
      labels = {
        Environment = var.environment
        NodeType    = "general"
      }
      
      taints = []
      
      tags = {
        NodeGroup = "general"
      }
    }
    
    # High-performance nodes for ML workloads
    ml = {
      name = "ml-nodes"
      
      min_size     = 1
      max_size     = 5
      desired_size = 2
      
      instance_types = ["g4dn.xlarge"]  # GPU instances
      capacity_type  = "ON_DEMAND"
      
      labels = {
        Environment = var.environment
        NodeType    = "ml"
        Accelerator = "nvidia-tesla-t4"
      }
      
      taints = [
        {
          key    = "ml-workload"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = {
        NodeGroup = "ml"
      }
    }
    
    # Critical system nodes
    critical = {
      name = "critical-nodes"
      
      min_size     = 3
      max_size     = 6
      desired_size = 3
      
      instance_types = ["m5.2xlarge"]
      capacity_type  = "ON_DEMAND"
      
      labels = {
        Environment = var.environment
        NodeType    = "critical"
      }
      
      taints = [
        {
          key    = "critical"
          value  = "true"
          effect = "NO_SCHEDULE"
        }
      ]
      
      tags = {
        NodeGroup = "critical"
      }
    }
  }

  # OIDC Provider
  enable_irsa = true

  # Cluster access
  manage_aws_auth_configmap = true
  aws_auth_roles = [
    {
      rolearn  = aws_iam_role.admin.arn
      username = "admin"
      groups   = ["system:masters"]
    },
  ]

  tags = local.common_tags
}

# KMS Key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS Secret Encryption Key"
  deletion_window_in_days = 10
  enable_key_rotation     = true

  tags = merge(
    local.common_tags,
    {
      Name = "${local.cluster_name}-eks-key"
    }
  )
}

resource "aws_kms_alias" "eks" {
  name          = "alias/${local.cluster_name}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# Additional security group for nodes
resource "aws_security_group" "node_group_additional" {
  name_prefix = "${local.cluster_name}-node-additional-"
  vpc_id      = module.vpc.vpc_id

  ingress {
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
    cidr_blocks = [
      "10.0.0.0/8",
    ]
  }

  tags = merge(
    local.common_tags,
    {
      Name = "${local.cluster_name}-node-additional"
    }
  )
}

# IAM Role for administrators
resource "aws_iam_role" "admin" {
  name = "${local.cluster_name}-admin"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          AWS = "arn:aws:iam::${data.aws_caller_identity.current.account_id}:root"
        }
      }
    ]
  })

  tags = local.common_tags
}

# RDS for MongoDB replacement (DocumentDB)
resource "aws_docdb_cluster" "main" {
  cluster_identifier      = "${local.cluster_name}-docdb"
  engine                  = "docdb"
  master_username         = "a2zadmin"
  master_password         = var.db_password
  backup_retention_period = 7
  preferred_backup_window = "07:00-09:00"
  skip_final_snapshot     = false
  final_snapshot_identifier = "${local.cluster_name}-docdb-final-${formatdate("YYYY-MM-DD-hhmm", timestamp())}"
  
  db_subnet_group_name = aws_docdb_subnet_group.main.name
  
  vpc_security_group_ids = [aws_security_group.docdb.id]
  
  enabled_cloudwatch_logs_exports = ["audit", "profiler"]
  
  tags = local.common_tags
}

resource "aws_docdb_cluster_instance" "main" {
  count              = 3
  identifier         = "${local.cluster_name}-docdb-${count.index}"
  cluster_identifier = aws_docdb_cluster.main.id
  instance_class     = "db.r5.large"
  
  tags = local.common_tags
}

resource "aws_docdb_subnet_group" "main" {
  name       = "${local.cluster_name}-docdb-subnet"
  subnet_ids = module.vpc.private_subnets
  
  tags = local.common_tags
}

resource "aws_security_group" "docdb" {
  name_prefix = "${local.cluster_name}-docdb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 27017
    to_port         = 27017
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  tags = local.common_tags
}

# ElastiCache for Redis
resource "aws_elasticache_replication_group" "main" {
  replication_group_id       = "${local.cluster_name}-redis"
  description               = "Redis cluster for A2Z TSN"
  node_type                 = "cache.r6g.large"
  port                      = 6379
  parameter_group_name      = aws_elasticache_parameter_group.redis.name
  subnet_group_name         = aws_elasticache_subnet_group.main.name
  security_group_ids        = [aws_security_group.redis.id]
  
  snapshot_retention_limit  = 5
  snapshot_window          = "03:00-05:00"
  
  at_rest_encryption_enabled = true
  transit_encryption_enabled = true
  auth_token                = var.redis_auth_token
  
  automatic_failover_enabled = true
  multi_az_enabled          = true
  num_cache_clusters        = 3
  
  tags = local.common_tags
}

resource "aws_elasticache_parameter_group" "redis" {
  family = "redis7"
  name   = "${local.cluster_name}-redis-params"
  
  parameter {
    name  = "maxmemory-policy"
    value = "allkeys-lru"
  }
  
  tags = local.common_tags
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${local.cluster_name}-redis-subnet"
  subnet_ids = module.vpc.private_subnets
  
  tags = local.common_tags
}

resource "aws_security_group" "redis" {
  name_prefix = "${local.cluster_name}-redis-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port       = 6379
    to_port         = 6379
    protocol        = "tcp"
    security_groups = [module.eks.node_security_group_id]
  }
  
  tags = local.common_tags
}

# S3 Buckets
resource "aws_s3_bucket" "data" {
  bucket = "${local.cluster_name}-data-${data.aws_caller_identity.current.account_id}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_encryption" "data" {
  bucket = aws_s3_bucket.data.id
  
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket" "backups" {
  bucket = "${local.cluster_name}-backups-${data.aws_caller_identity.current.account_id}"
  
  tags = local.common_tags
}

resource "aws_s3_bucket_lifecycle_configuration" "backups" {
  bucket = aws_s3_bucket.backups.id
  
  rule {
    id = "backup-retention"
    
    expiration {
      days = 90
    }
    
    transition {
      days          = 30
      storage_class = "STANDARD_IA"
    }
    
    transition {
      days          = 60
      storage_class = "GLACIER"
    }
    
    status = "Enabled"
  }
}

# ECR Repositories
resource "aws_ecr_repository" "app" {
  for_each = toset([
    "a2z-main",
    "a2z-ml",
    "a2z-backup",
    "a2z-failover"
  ])
  
  name                 = each.value
  image_tag_mutability = "MUTABLE"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  encryption_configuration {
    encryption_type = "AES256"
  }
  
  tags = local.common_tags
}

resource "aws_ecr_lifecycle_policy" "app" {
  for_each   = aws_ecr_repository.app
  repository = each.value.name
  
  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Keep last 10 images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v"]
          countType     = "imageCountMoreThan"
          countNumber   = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# CloudWatch Log Groups
resource "aws_cloudwatch_log_group" "application" {
  name              = "/aws/eks/${local.cluster_name}/application"
  retention_in_days = 30
  
  tags = local.common_tags
}

# Application Load Balancer
resource "aws_lb" "main" {
  name               = "${local.cluster_name}-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets           = module.vpc.public_subnets
  
  enable_deletion_protection = true
  enable_http2              = true
  enable_cross_zone_load_balancing = true
  
  tags = local.common_tags
}

resource "aws_security_group" "alb" {
  name_prefix = "${local.cluster_name}-alb-"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  tags = local.common_tags
}

# WAF for ALB
resource "aws_wafv2_web_acl" "main" {
  name  = "${local.cluster_name}-waf"
  scope = "REGIONAL"
  
  default_action {
    allow {}
  }
  
  rule {
    name     = "RateLimitRule"
    priority = 1
    
    statement {
      rate_based_statement {
        limit              = 2000
        aggregate_key_type = "IP"
      }
    }
    
    action {
      block {}
    }
    
    visibility_config {
      cloudwatch_metrics_enabled = true
      metric_name                = "RateLimitRule"
      sampled_requests_enabled   = true
    }
  }
  
  visibility_config {
    cloudwatch_metrics_enabled = true
    metric_name                = "${local.cluster_name}-waf"
    sampled_requests_enabled   = true
  }
  
  tags = local.common_tags
}

resource "aws_wafv2_web_acl_association" "main" {
  resource_arn = aws_lb.main.arn
  web_acl_arn  = aws_wafv2_web_acl.main.arn
}

# Route53 Hosted Zone
resource "aws_route53_zone" "main" {
  count = var.create_route53_zone ? 1 : 0
  name  = var.domain_name
  
  tags = local.common_tags
}

# ACM Certificate
resource "aws_acm_certificate" "main" {
  count             = var.create_route53_zone ? 1 : 0
  domain_name       = var.domain_name
  validation_method = "DNS"
  
  subject_alternative_names = [
    "*.${var.domain_name}"
  ]
  
  lifecycle {
    create_before_destroy = true
  }
  
  tags = local.common_tags
}

resource "aws_route53_record" "cert_validation" {
  for_each = var.create_route53_zone ? {
    for dvo in aws_acm_certificate.main[0].domain_validation_options : dvo.domain_name => {
      name   = dvo.resource_record_name
      record = dvo.resource_record_value
      type   = dvo.resource_record_type
    }
  } : {}
  
  allow_overwrite = true
  name            = each.value.name
  records         = [each.value.record]
  ttl             = 60
  type            = each.value.type
  zone_id         = aws_route53_zone.main[0].zone_id
}

# Outputs
output "cluster_endpoint" {
  description = "Endpoint for EKS control plane"
  value       = module.eks.cluster_endpoint
}

output "cluster_security_group_id" {
  description = "Security group ID attached to the EKS cluster"
  value       = module.eks.cluster_security_group_id
}

output "cluster_name" {
  description = "Kubernetes Cluster Name"
  value       = module.eks.cluster_name
}

output "cluster_certificate_authority_data" {
  description = "Base64 encoded certificate data required to communicate with the cluster"
  value       = module.eks.cluster_certificate_authority_data
  sensitive   = true
}

output "ecr_repositories" {
  description = "ECR repository URLs"
  value       = { for k, v in aws_ecr_repository.app : k => v.repository_url }
}

output "alb_dns" {
  description = "DNS name of the load balancer"
  value       = aws_lb.main.dns_name
}

output "docdb_endpoint" {
  description = "DocumentDB cluster endpoint"
  value       = aws_docdb_cluster.main.endpoint
}

output "redis_endpoint" {
  description = "Redis primary endpoint"
  value       = aws_elasticache_replication_group.main.primary_endpoint_address
}