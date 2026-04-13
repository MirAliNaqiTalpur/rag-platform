terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.25"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

data "google_project" "current" {
  project_id = var.project_id
}

locals {
  required_services = [
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com",
    "storage.googleapis.com",
    "iam.googleapis.com",
    "secretmanager.googleapis.com",
    "logging.googleapis.com",
    "monitoring.googleapis.com"
  ]
}

resource "google_project_service" "services" {
  for_each                   = toset(local.required_services)
  project                    = var.project_id
  service                    = each.value
  disable_on_destroy         = false
  disable_dependent_services = false
}

resource "google_artifact_registry_repository" "repo" {
  provider      = google
  project       = var.project_id
  location      = var.region
  repository_id = var.artifact_repo_name
  description   = "Container images for modular RAG platform"
  format        = "DOCKER"

  depends_on = [google_project_service.services]
}

resource "google_storage_bucket" "documents" {
  name                        = var.bucket_name
  location                    = var.region
  project                     = var.project_id
  storage_class               = "STANDARD"
  uniform_bucket_level_access = true
  force_destroy               = false

  versioning {
    enabled = true
  }

  depends_on = [google_project_service.services]
}

resource "google_service_account" "runtime" {
  account_id   = "rag-platform-runtime"
  display_name = "RAG Platform Runtime Service Account"
  project      = var.project_id

  depends_on = [google_project_service.services]
}

resource "google_storage_bucket_iam_member" "runtime_bucket_access" {
  bucket = google_storage_bucket.documents.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.runtime.email}"
}

resource "google_cloud_run_v2_service" "rag_engine" {
  name                = var.rag_service_name
  location            = var.region
  project             = var.project_id
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email
    timeout         = "300s"

    containers {
      image = var.rag_container_image

      ports {
        container_port = 8001
      }

      env {
        name  = "VECTOR_STORE"
        value = var.vector_store
      }

      env {
        name  = "RETRIEVER"
        value = var.retriever
      }

      env {
        name  = "RERANKER"
        value = var.reranker
      }

      env {
        name  = "GENERATOR"
        value = var.generator
      }

      env {
        name  = "DOCUMENT_SOURCE"
        value = var.document_source
      }

      env {
        name  = "TOP_K"
        value = tostring(var.top_k)
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GCS_BUCKET_NAME"
        value = google_storage_bucket.documents.name
      }

      env {
        name  = "GCS_PREFIX"
        value = var.documents_prefix
      }

      env {
        name  = "FAISS_INDEX_PREFIX"
        value = var.faiss_index_prefix
      }

      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_storage_bucket.documents,
    google_service_account.runtime
  ]
}

resource "google_cloud_run_v2_service_iam_member" "rag_public" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.rag_engine.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service" "mcp_server" {
  name                = var.mcp_service_name
  location            = var.region
  project             = var.project_id
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email
    timeout         = "300s"

    containers {
      image = var.mcp_container_image

      ports {
        container_port = 8000
      }

      env {
        name  = "RAG_BASE_URL"
        value = google_cloud_run_v2_service.rag_engine.uri
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "512Mi"
        }
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_cloud_run_v2_service.rag_engine
  ]
}

resource "google_cloud_run_v2_service_iam_member" "mcp_public" {
  count    = var.allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.mcp_server.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service" "streamlit_ui" {
  count               = var.deploy_ui ? 1 : 0
  name                = var.ui_service_name
  location            = var.region
  project             = var.project_id
  ingress             = "INGRESS_TRAFFIC_ALL"
  deletion_protection = false

  template {
    service_account = google_service_account.runtime.email
    timeout         = "300s"

    containers {
      image = var.ui_container_image

      ports {
        container_port = 8501
      }

      env {
        name  = "VECTOR_STORE"
        value = var.vector_store
      }

      env {
        name  = "RETRIEVER"
        value = var.retriever
      }

      env {
        name  = "RERANKER"
        value = var.reranker
      }

      env {
        name  = "GENERATOR"
        value = var.generator
      }

      env {
        name  = "DOCUMENT_SOURCE"
        value = var.document_source
      }

      env {
        name  = "TOP_K"
        value = tostring(var.top_k)
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GCP_PROJECT"
        value = var.project_id
      }

      env {
        name  = "GCS_BUCKET_NAME"
        value = google_storage_bucket.documents.name
      }

      env {
        name  = "GCS_PREFIX"
        value = var.documents_prefix
      }

      env {
        name  = "FAISS_INDEX_PREFIX"
        value = var.faiss_index_prefix
      }

      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }

      env {
        name  = "RAG_API_URL"
        value = google_cloud_run_v2_service.rag_engine.uri
      }

      env {
        name  = "STREAMLIT_SERVER_FILE_WATCHER_TYPE"
        value = "none"
      }

      resources {
        limits = {
          cpu    = "1"
          memory = "1Gi"
        }
      }
    }
  }

  depends_on = [
    google_project_service.services,
    google_cloud_run_v2_service.rag_engine
  ]
}

resource "google_cloud_run_v2_service_iam_member" "ui_public" {
  count    = var.deploy_ui && var.allow_unauthenticated ? 1 : 0
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.streamlit_ui[0].name
  role     = "roles/run.invoker"
  member   = "allUsers"
}