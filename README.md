# Modular MCP-Enabled RAG Platform

> A cloud-agnostic, modular RAG infrastructure platform with MCP integration, designed for reproducible deployment using Docker and Terraform.

A production-oriented, modular Retrieval-Augmented Generation (RAG) platform with a Model Context Protocol (MCP) server interface, pluggable retrieval and reranking strategies, interchangeable vector backends, and Terraform-based cloud deployment.

---

## Project overview

This repository implements a cloud-agnostic RAG infrastructure layer designed for reusable deployment across document-centric AI systems. The platform is built around four separable concerns:

- **MCP server** for protocol-based tool exposure to LLM agents
- **RAG engine** for retrieval, reranking, and answer generation orchestration
- **Vector storage abstraction** for switching between vector backends
- **Evaluation-ready structure** for reproducible experimentation

The current implementation is validated for:

- local execution
- Docker Compose deployment
- GCP Cloud Run deployment using Terraform
- GCS-backed document loading and runtime dataset switching
- GCS-backed FAISS index persistence

---

## Current validated progress vs proposal milestones

**Status:** Month 1 and Month 2 milestone goals have been implemented and validated. Month 3 evaluation is pending.

### Month 1: Core architecture — completed

- modular separation of services
- Docker-based local setup
- Terraform-based GCP deployment
- configuration-driven behavior
- cloud portability (local → Docker → Cloud Run)

### Month 2: MCP + modular RAG — completed

- MCP tool exposure
- pluggable retrieval + reranking
- runtime dataset reload
- GCS-backed document loading
- vector backend switching (FAISS / Chroma)

### Month 3: Evaluation — pending

- Recall@K, Precision@K, MRR
- strategy comparison
- experiment tracking

---

## Architecture overview

### Services

1. **rag-engine**
   - loads documents from local or GCS
   - builds vector index
   - serves `/query`, `/search`, `/models`, `/reload-dataset`, and `/restore-index`

2. **mcp-server**
   - exposes retrieval tools via MCP
   - delegates to rag-engine

3. **streamlit-ui**
   - UI for querying, search, dataset reload
   - supports default GCS bucket with override

4. **storage layer**
   - GCS for documents
   - FAISS index persisted to GCS
   - Chroma backend for modular switching

---

## Vector storage and persistence

The platform supports interchangeable vector backends through a unified abstraction layer.

### FAISS (primary backend)

- used for cloud deployment
- index artifacts stored in GCS
- enables stateless Cloud Run services

Stored at:

```text
gs://<bucket>/indexes/faiss/
```

Artifacts:

- `index.faiss`
- `docs.json`

### Chroma (secondary backend)

- used to demonstrate backend modularity
- switchable via configuration
- not the primary cloud persistence backend

### Persistence flow

1. Documents loaded from:

```text
gs://<bucket>/documents/
```

2. FAISS index is built
3. Index is saved locally inside the container
4. Index artifacts are uploaded to:

```text
gs://<bucket>/indexes/faiss/
```

5. The index can be restored through the API

---

## Restore endpoint

Restore the FAISS index from GCS:

```bash
curl -X POST <RAG_URL>/restore-index
```

Expected response:

```json
{
  "status": "success",
  "restored": true
}
```

---

## Repository structure

```text
app/
  core/
  ingestion/
  mcp/
  rag/
  vectorstore/
  rag_api.py
  ui.py
infra/
  terraform/
docker/
data/
  documents/
```

---

## Features

- modular RAG pipeline
- MCP integration
- FAISS + Chroma backend switching
- GCS-backed documents
- FAISS persistence to GCS
- runtime dataset reload
- Terraform deployment
- Streamlit UI

---

## Configuration

### Core variables

- `VECTOR_STORE`
- `RETRIEVER`
- `RERANKER`
- `GENERATOR`
- `DOCUMENT_SOURCE`
- `TOP_K`

### GCS / persistence variables

- `GCS_BUCKET_NAME`
- `GCS_PREFIX`
- `INDEX_STORAGE`
- `GCS_INDEX_BUCKET`
- `GCS_INDEX_PREFIX`

---

## Local development

Clone the repository:

```bash
git clone https://github.com/MirAliNaqiTalpur/rag-platform.git
cd rag-platform
```

Create a local environment file:

```bash
cp .env.example .env.local
```

Set at least:

```env
GEMINI_API_KEY=your_key
VECTOR_STORE=faiss
DOCUMENT_SOURCE=local
```

Run locally with Docker Compose:

```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## Cloud deployment (GCP)

### Build and push images

From the repository root:

```bash
docker build --no-cache -f docker/Dockerfile.rag -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/rag-engine:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/rag-engine:latest

docker build --no-cache -f docker/Dockerfile.mcp -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/mcp-server:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/mcp-server:latest

docker build --no-cache -f docker/Dockerfile.ui -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/streamlit-ui:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/streamlit-ui:latest
```

### Terraform setup

Create:

```bash
infra/terraform/terraform.tfvars
```

Example:

```hcl
project_id = "your-project-id"
region     = "asia-southeast1"

artifact_repo_name = "rag-platform-repo"

rag_service_name = "rag-engine"
mcp_service_name = "mcp-server"
ui_service_name  = "streamlit-ui"

bucket_name         = "your-bucket"
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
```

### Deploy

Run validation before applying infrastructure:

```bash
cd infra/terraform
terraform init
terraform validate
terraform plan
terraform apply
```

---

## Post-deployment validation checklist

Verify the following after deployment:

- `rag-engine` `/health` works
- UI loads successfully
- reload dataset works
- GCS contains:
  - `documents/`
  - `indexes/faiss/`
- `/restore-index` works
- FAISS query works
- Chroma backend can be selected and queried

---

## Destroying cloud infrastructure

To simulate a clean delivery and redeployment cycle:

```bash
cd infra/terraform
terraform destroy
```

---

## Limitations

- UI is minimal and operational, not a production front-end
- Chroma is not currently used as the primary persistent cloud backend
- the evaluation framework is not yet fully implemented

---

## Final note

This system demonstrates a modular, cloud-agnostic RAG architecture where:

- core logic is independent of cloud provider
- infrastructure is provisioned via Terraform
- vector index persistence is externalized from the container lifecycle

This satisfies the internship objective of building a reusable RAG infrastructure layer.
