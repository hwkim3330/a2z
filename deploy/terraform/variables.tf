# Terraform Variables for A2Z TSN/FRER Infrastructure

variable "aws_region" {
  description = "AWS region for deployment"
  type        = string
  default     = "ap-northeast-2"  # Seoul
}

variable "environment" {
  description = "Environment name (dev, staging, production)"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "production"], var.environment)
    error_message = "Environment must be dev, staging, or production."
  }
}

variable "kubernetes_version" {
  description = "Kubernetes version for EKS"
  type        = string
  default     = "1.28"
}

variable "db_password" {
  description = "Password for DocumentDB"
  type        = string
  sensitive   = true
}

variable "redis_auth_token" {
  description = "Auth token for Redis"
  type        = string
  sensitive   = true
}

variable "domain_name" {
  description = "Domain name for the application"
  type        = string
  default     = "a2z-tsn.com"
}

variable "create_route53_zone" {
  description = "Whether to create Route53 hosted zone"
  type        = bool
  default     = true
}

variable "monitoring_enabled" {
  description = "Enable CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "backup_enabled" {
  description = "Enable automated backups"
  type        = bool
  default     = true
}

variable "high_availability" {
  description = "Enable high availability mode"
  type        = bool
  default     = true
}

variable "enable_gpu_nodes" {
  description = "Enable GPU nodes for ML workloads"
  type        = bool
  default     = true
}

variable "node_groups" {
  description = "Configuration for EKS node groups"
  type = map(object({
    min_size       = number
    max_size       = number
    desired_size   = number
    instance_types = list(string)
    capacity_type  = string
    labels         = map(string)
    taints = list(object({
      key    = string
      value  = string
      effect = string
    }))
  }))
  default = {}
}

variable "vpc_cidr" {
  description = "CIDR block for VPC"
  type        = string
  default     = "10.0.0.0/16"
}

variable "allowed_cidr_blocks" {
  description = "CIDR blocks allowed to access the cluster"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "tags" {
  description = "Additional tags for resources"
  type        = map(string)
  default     = {}
}