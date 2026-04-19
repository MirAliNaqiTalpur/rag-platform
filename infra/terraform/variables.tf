variable "project_id" {
  description = "GCP project ID for deployment"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "asia-southeast1"
}

variable "artifact_repo_name" {
  description = "Artifact Registry repository name"
  type        = string
  default     = "rag-platform-repo"
}

variable "rag_service_name" {
  description = "Cloud Run service name for the RAG engine"
  type        = string
  default     = "rag-engine"
}

variable "mcp_service_name" {
  description = "Cloud Run service name for the MCP server"
  type        = string
  default     = "mcp-server"
}

variable "bucket_name" {
  description = "Cloud Storage bucket for documents and FAISS index artifacts"
  type        = string
}

variable "documents_prefix" {
  description = "GCS prefix for source documents"
  type        = string
  default     = "documents"
}

variable "faiss_index_prefix" {
  description = "GCS prefix for FAISS index artifacts"
  type        = string
  default     = "indexes/faiss"
}

variable "rag_container_image" {
  description = "Full container image URI for rag-engine"
  type        = string
}

variable "mcp_container_image" {
  description = "Full container image URI for mcp-server"
  type        = string
}

variable "vector_store" {
  description = "Vector store backend"
  type        = string
  default     = "faiss"
}

variable "retriever" {
  description = "Retriever strategy"
  type        = string
  default     = "hybrid"
}

variable "reranker" {
  description = "Reranker strategy"
  type        = string
  default     = "simple"
}

variable "generator" {
  description = "Generator backend"
  type        = string
  default     = "gemini"
}

variable "document_source" {
  description = "Document source backend"
  type        = string
  default     = "gcs"
}

variable "top_k" {
  description = "Top K retrieval count"
  type        = number
  default     = 3
}

variable "gemini_secret_name" {
  description = "Secret Manager secret name containing the Gemini API key"
  type        = string
  default     = "gemini-api-key"
}

variable "allow_unauthenticated" {
  description = "Allow public unauthenticated access to Cloud Run services"
  type        = bool
  default     = true
}

variable "ui_service_name" {
  description = "Cloud Run service name for the Streamlit UI"
  type        = string
  default     = "streamlit-ui"
}

variable "ui_container_image" {
  description = "Full container image URI for streamlit-ui"
  type        = string
}

variable "deploy_ui" {
  description = "Whether to deploy the Streamlit UI service"
  type        = bool
  default     = true
}

variable "allow_custom_models" {
  description = "Whether to allow custom model names in the UI and backend"
  type        = bool
  default     = true
}

variable "deploy_services" {
  description = "Whether to deploy Cloud Run services"
  type        = bool
  default     = false
}