# Modular MCP-Enabled RAG Platform

> A cloud-agnostic, modular RAG infrastructure platform with MCP integration, designed for reproducible deployment using Docker and Terraform.

A production-oriented, modular Retrieval-Augmented Generation (RAG) platform with a Model Context Protocol (MCP) server interface, pluggable retrieval and reranking strategies, interchangeable vector backends, and Terraform-based cloud deployment.

---

## Project overview

This repository implements a cloud-agnostic RAG infrastructure layer designed for reusable deployment across document-centric AI systems. The platform is built around four separable concerns:

* **MCP server** for protocol-based tool exposure to LLM agents
* **RAG engine** for retrieval, reranking, and answer generation orchestration
* **Vector storage abstraction** for switching between vector backends
* **Evaluation-ready structure** for reproducible experimentation

The current implementation is validated for:

* local execution
* Docker Compose deployment
* GCP Cloud Run deployment using Terraform
* GCS-backed document loading and runtime dataset switching
* GCS-backed FAISS index persistence

---

## Current validated progress vs proposal milestones

**Status:** Month 1 and Month 2 milestone goals have been implemented and validated. Month 3 evaluation is pending.

### Month 1: Core architecture — completed

* modular separation of services
* Docker-based local setup
* Terraform-based GCP deployment
* configuration-driven behavior
* cloud portability (local → Docker → Cloud Run)

### Month 2: MCP + modular RAG — completed

* MCP tool exposure
* pluggable retrieval + reranking
* runtime dataset reload
* GCS-backed document loading
* vector backend switching (FAISS / Chroma)

### Month 3: Evaluation — pending

* Recall@K, Precision@K, MRR
* strategy comparison
* experiment tracking

---

## Architecture overview

### Services

1. **rag-engine**

   * loads documents from local or GCS
   * builds vector index
   * serves `/query`, `/search`, `/models`, `/reload-dataset`, and `/restore-index`

2. **mcp-server**

   * exposes retrieval tools via MCP
   * delegates to rag-engine

3. **streamlit-ui**

   * UI for querying, search, dataset reload

4. **storage layer**

   * GCS for documents
   * FAISS index persisted to GCS
   * Chroma backend for modular switching

---

## Vector storage and persistence

### FAISS (primary backend)

* used for cloud deployment
* index artifacts stored in GCS
* enables stateless Cloud Run services

Stored at:

```
gs://<bucket>/indexes/faiss/
```

Artifacts:

* `index.faiss`
* `docs.json`

### Chroma (secondary backend)

* used to demonstrate backend modularity
* switchable via configuration

### Persistence flow

1. Documents loaded from:

```
gs://<bucket>/documents/
```

2. FAISS index is built
3. Index is saved locally
4. Index uploaded to:

```
gs://<bucket>/indexes/faiss/
```

5. Can be restored via API

---

## Restore endpoint

```bash
curl -X POST <RAG_URL>/restore-index
```

---

## Repository structure

```
app/
infra/
docker/
data/
```

---

## Features

* modular RAG pipeline
* MCP integration
* FAISS + Chroma switching
* GCS-backed documents
* FAISS persistence
* runtime dataset reload
* Terraform deployment
* Streamlit UI

---

## Configuration

### Core variables

* VECTOR_STORE
* RETRIEVER
* RERANKER
* GENERATOR
* DOCUMENT_SOURCE
* TOP_K

### GCS variables

* GCS_BUCKET_NAME
* GCS_PREFIX
* GCS_INDEX_PREFIX

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

## Terraform-first deployment

### Create Gemini secret

```bash
echo -n "YOUR_GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-
```

---

### Create terraform.tfvars

```bash
infra/terraform/terraform.tfvars
```

```hcl
project_id = "your-project-id"
region     = "asia-southeast1"

artifact_repo_name = "rag-platform-repo"

rag_service_name = "rag-engine"
mcp_service_name = "mcp-server"
ui_service_name  = "streamlit-ui"

bucket_name = "your-bucket"

rag_container_image = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/rag-engine:latest"
mcp_container_image = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/mcp-server:latest"
ui_container_image  = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/streamlit-ui:latest"

gemini_secret_name = "gemini-api-key"
```

---

### First Terraform apply

```bash
cd infra/terraform
terraform init
terraform apply
```

Creates:

* Artifact Registry
* GCS
* IAM
* Cloud Run

---

### Configure Docker

```bash
gcloud auth configure-docker asia-southeast1-docker.pkg.dev
```

---

### Build & push images

```bash
docker build -f docker/Dockerfile.rag -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/rag-engine:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/rag-platform-repo/rag-engine:latest
```

(repeat for mcp + ui)

---

### Second Terraform apply

```bash
terraform apply
```

---

## Validation checklist

* rag-engine `/health` works
* UI loads
* dataset reload works
* GCS has documents + indexes
* restore-index works

---

## Destroy

```bash
terraform destroy
```

---

## Final note

This project demonstrates:

* modular RAG design
* MCP-based tool exposure
* Terraform-based deployment
* cloud-agnostic architecture

Designed as a **production-style internship deliverable**.
