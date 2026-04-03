variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-southeast1"
}

variable "mcp_service_name" {
  description = "Cloud Run MCP service name"
  type        = string
  default     = "mcp-server"
}

variable "rag_service_name" {
  description = "Cloud Run RAG service name"
  type        = string
  default     = "rag-engine"
}

variable "artifact_repo_name" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "rag-platform-repo"
}

variable "mcp_container_image" {
  description = "Full Artifact Registry image URL for MCP service"
  type        = string
}

variable "rag_container_image" {
  description = "Full Artifact Registry image URL for RAG service"
  type        = string
}

variable "generator" {
  description = "Generator mode"
  type        = string
  default     = "mock"
}

variable "gemini_api_key" {
  description = "Gemini API key"
  type        = string
  sensitive   = true
  default     = ""
}