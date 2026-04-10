output "artifact_registry_repository" {
  description = "Artifact Registry repository name"
  value       = google_artifact_registry_repository.repo.repository_id
}

output "artifact_registry_location" {
  description = "Artifact Registry repository location"
  value       = google_artifact_registry_repository.repo.location
}

output "gcs_bucket_name" {
  description = "Cloud Storage bucket used for documents and FAISS artifacts"
  value       = google_storage_bucket.documents.name
}

output "runtime_service_account_email" {
  description = "Service account used by Cloud Run services"
  value       = google_service_account.runtime.email
}

output "rag_service_url" {
  description = "URL of the RAG engine Cloud Run service"
  value       = google_cloud_run_v2_service.rag_engine.uri
}

output "mcp_service_url" {
  description = "URL of the MCP server Cloud Run service"
  value       = google_cloud_run_v2_service.mcp_server.uri
}

output "streamlit_ui_url" {
  description = "URL of the Streamlit UI Cloud Run service"
  value       = var.deploy_ui ? google_cloud_run_v2_service.streamlit_ui[0].uri : null
}