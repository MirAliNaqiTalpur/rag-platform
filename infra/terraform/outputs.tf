output "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.repo.repository_id
}

output "artifact_registry_location" {
  description = "Artifact Registry repository location"
  value       = google_artifact_registry_repository.repo.location
}

output "bucket_name" {
  description = "GCS bucket for documents and FAISS index"
  value       = google_storage_bucket.documents.name
}

output "service_account_email" {
  description = "Runtime service account email"
  value       = google_service_account.runtime.email
}

output "rag_service_url" {
  description = "URL of the RAG engine Cloud Run service"
  value       = var.deploy_services ? google_cloud_run_v2_service.rag_engine[0].uri : null
}

output "mcp_service_url" {
  description = "URL of the MCP server Cloud Run service"
  value       = var.deploy_services ? google_cloud_run_v2_service.mcp_server[0].uri : null
}

output "streamlit_ui_url" {
  description = "URL of the Streamlit UI Cloud Run service"
  value       = var.deploy_services && var.deploy_ui ? google_cloud_run_v2_service.streamlit_ui[0].uri : null
}