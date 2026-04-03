terraform {
  required_version = ">= 1.5.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 7.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

resource "google_project_service" "required" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "cloudbuild.googleapis.com"
  ])

  project = var.project_id
  service = each.value

  disable_on_destroy = false
}

resource "google_artifact_registry_repository" "docker_repo" {
  location      = var.region
  repository_id = var.artifact_repo_name
  description   = "Docker repository for rag platform"
  format        = "DOCKER"

  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service" "rag_engine" {
  name     = var.rag_service_name
  location = var.region

  deletion_protection = false

  template {
    containers {
      image = var.rag_container_image

      ports {
        container_port = 8001
      }

      resources {
        limits = {
          memory = "1024Mi"
        }
      }

      env {
        name  = "GENERATOR"
        value = var.generator
      }

      env {
        name  = "PYTHONPATH"
        value = "/app"
      }

      env {
        name  = "GEMINI_API_KEY"
        value = var.gemini_api_key
      }
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service" "mcp_server" {
  name     = var.mcp_service_name
  location = var.region

  deletion_protection = false

  template {
    containers {
      image = var.mcp_container_image

      ports {
        container_port = 8000
      }

      resources {
        limits = {
          memory = "1024Mi"
        }
      }

      env {
        name  = "PYTHONPATH"
        value = "/app"
      }

      env {
        name  = "RAG_BASE_URL"
        value = google_cloud_run_v2_service.rag_engine.uri
      }
    }
  }

  depends_on = [google_project_service.required]
}

resource "google_cloud_run_v2_service_iam_member" "mcp_public_access" {
  location = google_cloud_run_v2_service.mcp_server.location
  name     = google_cloud_run_v2_service.mcp_server.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

resource "google_cloud_run_v2_service_iam_member" "rag_public_access" {
  location = google_cloud_run_v2_service.rag_engine.location
  name     = google_cloud_run_v2_service.rag_engine.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

output "mcp_service_name" {
  value = google_cloud_run_v2_service.mcp_server.name
}

output "mcp_service_url" {
  value = google_cloud_run_v2_service.mcp_server.uri
}

output "rag_service_name" {
  value = google_cloud_run_v2_service.rag_engine.name
}

output "rag_service_url" {
  value = google_cloud_run_v2_service.rag_engine.uri
}