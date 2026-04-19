# Modular MCP-Enabled RAG Platform

> A cloud-agnostic, modular RAG infrastructure platform with MCP integration, designed for reproducible deployment using Docker and Terraform.

A production-oriented, modular Retrieval-Augmented Generation (RAG) platform with a Model Context Protocol (MCP) server interface, pluggable retrieval and reranking strategies, interchangeable vector backends, and Terraform-based cloud deployment.

---

## Project overview

This repository implements a cloud-agnostic RAG infrastructure layer designed for reusable deployment across document-centric AI systems.

The platform is built around four separable concerns:

* **MCP server** for protocol-based tool exposure
* **RAG engine** for retrieval, reranking, and generation
* **Vector storage abstraction** for backend switching
* **Evaluation-ready structure** for experimentation

Validated for:

* local execution
* Docker Compose deployment
* Terraform-based Cloud Run deployment
* GCS-backed document loading
* FAISS persistence on GCS

---

## Architecture overview

### Services

1. **rag-engine**

   * loads documents from local or GCS
   * builds vector index
   * serves `/query`, `/search`, `/models`, `/reload-dataset`, `/restore-index`

2. **mcp-server**

   * exposes tools via MCP
   * calls rag-engine internally

3. **streamlit-ui**

   * Frontend interface for interacting with the RAG system

---

## Demo

The Streamlit interface provides an interactive front-end for submitting queries, viewing generated answers, and inspecting retrieved documents.

<p align="center">
  <img src="images/streamlit-ui-demo.png" alt="Streamlit UI Demo" width="900"/>
</p>

4. **storage**

   * GCS for documents + index
   * FAISS (primary)
   * Chroma (secondary)

---

## Vector storage

### FAISS (primary)

Stored in:

```
gs://<bucket>/indexes/faiss/
```

Artifacts:

* `index.faiss`
* `docs.json`

### Chroma (secondary)

* used for modular backend demonstration

---

## Local development

```bash
git clone https://github.com/MirAliNaqiTalpur/rag-platform.git
cd rag-platform
```

```bash
cp .env.example .env.local
```

Set:

```
GEMINI_API_KEY=your_key
VECTOR_STORE=faiss
DOCUMENT_SOURCE=local
```

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Cloud deployment (GCP)

### Project setup

```bash
gcloud projects create YOUR-PROJECT-ID
gcloud config set project YOUR-PROJECT-ID
```

### Link billing

```bash
gcloud billing projects link YOUR-PROJECT-ID --billing-account=YOUR-BILLING-ID
```

### Enable APIs

```bash
gcloud services enable \
  run.googleapis.com \
  artifactregistry.googleapis.com \
  cloudbuild.googleapis.com \
  secretmanager.googleapis.com \
  iam.googleapis.com
```

---

# Terraform-first deployment (FINAL FLOW)

## Overview

Deployment is done in **two phases**:

1. **Phase 1 — Infrastructure only**
2. **Phase 2 — Services deployment**

---

## Create Gemini secret

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
```

If exists:

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets versions add gemini-api-key --data-file=-
```

---

## Create terraform.tfvars

```bash
cd infra/terraform
cp terraform.tfvars.example terraform.tfvars
```

Edit:

```hcl
project_id = "your-project-id"
region     = "asia-southeast1"

artifact_repo_name = "rag-platform-repo"

rag_service_name = "rag-engine"
mcp_service_name = "mcp-server"
ui_service_name  = "streamlit-ui"

bucket_name         = "your-unique-bucket"
documents_prefix    = "documents"
faiss_index_prefix  = "indexes/faiss"

rag_container_image = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/rag-engine:latest"
mcp_container_image = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/mcp-server:latest"
ui_container_image  = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/streamlit-ui:latest"

vector_store    = "faiss"
retriever       = "hybrid"
reranker        = "simple"
generator       = "gemini"
document_source = "gcs"

top_k = 3

gemini_secret_name    = "gemini-api-key"
allow_unauthenticated = true
deploy_ui             = true
allow_custom_models   = true

deploy_services = false
```

---

## Phase 1 — Terraform apply (infra only)

```bash
terraform init
terraform validate
terraform plan
terraform apply
```

Creates:

* Artifact Registry
* GCS bucket
* IAM + service account
* Secret access

No Cloud Run services yet.

---

## Configure Docker

```bash
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
```

---

## Build and push images

```bash
docker build --no-cache -f docker/Dockerfile.rag -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/rag-engine:latest .
docker build --no-cache -f docker/Dockerfile.mcp -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/mcp-server:latest .
docker build --no-cache -f docker/Dockerfile.ui -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/streamlit-ui:latest .

docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/rag-engine:latest
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/mcp-server:latest
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/streamlit-ui:latest
```

---

## Phase 2 — Deploy services

Update:

```hcl
deploy_services = true
```

Then:

```bash
terraform plan
terraform apply
```

Outputs:

* `rag_service_url`
* `mcp_service_url`
* `streamlit_ui_url`

---

## Upload documents to GCS

```bash
gcloud storage cp -r data/documents/* gs://<BUCKET_NAME>/documents/
```

Verify:

```bash
gcloud storage ls gs://<BUCKET_NAME>/documents/
```

---

## Validation checklist

* Terraform outputs service URLs
* `/health` endpoint works
* UI loads
* documents uploaded to GCS
* dataset reload works
* index stored in `indexes/faiss/`
* `/restore-index` works
* queries return results

---

## Destroy

```bash
terraform destroy
```

---

## Final note

This project demonstrates:

* modular RAG architecture
* MCP integration
* Terraform-based deployment
* cloud-agnostic design

Designed as a **production-grade internship deliverable**.
