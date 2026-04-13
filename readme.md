# Modular MCP-Enabled RAG Platform

A production-oriented, modular Retrieval-Augmented Generation (RAG) platform with a Model Context Protocol (MCP) server interface, pluggable retrieval and reranking strategies, interchangeable vector backends, and Terraform-based cloud deployment.

## Project overview

This repository implements a cloud-agnostic RAG infrastructure layer designed for reusable deployment across document-centric AI systems. The platform is built around four separable concerns:

- **MCP server** for protocol-based tool exposure to LLM agents
- **RAG engine** for retrieval, reranking, and answer generation orchestration
- **Vector storage abstraction** for switching between local and cloud-backed document sources/backends
- **Evaluation-ready structure** for reproducible experimentation and comparison across configurations

The current implementation is validated for:

- local execution
- Docker Compose deployment
- GCP Cloud Run deployment using Terraform
- GCS-backed document loading and runtime dataset switching

The design goal is to keep core application logic cloud-agnostic while treating infrastructure providers as adapters.

## Internship alignment

This project was developed to satisfy the internship objective of designing and implementing a production-ready, modular RAG platform with MCP integration, backend portability, pluggable retrieval strategies, and deployment reproducibility.

### Month 1 milestone focus: Core architecture and infrastructure

Implemented and validated:

- service separation between MCP server, RAG engine, UI, and storage-backed document flow
- Docker-based local orchestration
- Terraform-based GCP deployment structure
- configuration-driven runtime behavior through environment variables
- local and GCS-backed document source switching
- backend-aligned Cloud Run configuration for deployed default GCS bucket usage

### Month 2 milestone focus: MCP integration and modular RAG components

Implemented and validated:

- MCP tool exposure for search and query workflows
- strategy-based retriever and reranker selection
- metadata-aware retrieval support
- runtime model selection support in the UI and backend
- cross-service validation across local, Docker, and Cloud Run

### Month 3 milestone focus: Evaluation and finalization

Planned / partially scaffolded:

- reproducible evaluation workflow
- metric reporting for retrieval quality
- experiment comparison across configurations
- deployment handoff documentation and onboarding workflow

## Current architecture

### Services

1. **rag-engine**
   - loads dataset from local documents or GCS
   - builds/rebuilds vector store
   - serves `/query`, `/search`, `/models`, `/health`, and `/reload-dataset`

2. **mcp-server**
   - exposes retrieval tools through MCP-compatible interfaces
   - delegates retrieval/query execution to rag-engine

3. **streamlit-ui**
   - provides a lightweight operator UI for querying, search-only retrieval, file upload, and dataset switching
   - supports default deployed GCS bucket usage with optional bucket override

4. **storage / indexing layer**
   - local documents under `data/documents`
   - GCS-backed document source for cloud deployment
   - FAISS-backed index for current retrieval flow

## Repository structure

```text
app/
  core/
  ingestion/
  mcp/
  rag/
  storage/
  vectorstore/
  rag_api.py
  ui.py
infra/
  terraform/
docker/
scripts/
data/
  documents/
```

## Features

- modular RAG engine with configurable retriever and reranker
- MCP server for tool-oriented access
- local and GCS-backed dataset loading
- runtime dataset reload from UI or API
- Terraform-managed Cloud Run deployment
- Docker Compose local development workflow
- configurable Gemini model selection
- backend-controlled default GCS bucket with optional UI override

## Configuration

### Core runtime variables

- `VECTOR_STORE`
- `RETRIEVER`
- `RERANKER`
- `GENERATOR`
- `DOCUMENT_SOURCE`
- `TOP_K`
- `DEFAULT_MODEL`
- `AVAILABLE_MODELS`
- `GEMINI_API_KEY`

### GCS / cloud deployment variables

- `GCS_BUCKET_NAME`
- `GCS_PREFIX`
- `FAISS_INDEX_PREFIX`
- `RAG_API_URL` for the Streamlit UI
- `RAG_BASE_URL` for MCP-to-RAG communication where applicable

## Local development

### 1. Clone the repository

```bash
git clone <your-repo-url>
cd rag-platform
```

### 2. Prepare environment

Create a local environment file as needed, for example:

```bash
cp .env.example .env.local
```

Set required values such as:

```env
GEMINI_API_KEY=your_key_here
VECTOR_STORE=faiss
RETRIEVER=hybrid
RERANKER=simple
GENERATOR=gemini
DOCUMENT_SOURCE=local
TOP_K=3
DEFAULT_MODEL=gemini-3.1-flash-lite-preview
AVAILABLE_MODELS=gemini-3.1-flash-lite-preview
```

### 3. Run locally with Docker Compose

```bash
docker compose -f docker/docker-compose.yml up --build
```

Typical service endpoints:

- UI: `http://localhost:8501`
- RAG API: `http://localhost:8001`
- MCP server: `http://localhost:8000`

## Local validation checklist

Validate the following before cloud deployment:

- UI loads successfully
- `/health` responds from rag-engine
- query endpoint returns grounded answers
- search-only retrieval returns documents
- switching `DOCUMENT_SOURCE` works
- local ingest and index rebuild complete without error
- MCP tool tests pass

## Cloud deployment on GCP

### Deployment model

The current cloud target uses:

- Artifact Registry for images
- Cloud Run for `rag-engine`, `mcp-server`, and `streamlit-ui`
- Cloud Storage for documents
- Terraform for reproducible infrastructure provisioning

### Pre-deployment prerequisites

- authenticated `gcloud` CLI
- enabled billing on the GCP project
- Docker installed and authenticated for Artifact Registry
- Terraform installed

### Build and push images

From the repository root:

```bash
docker build -f docker/Dockerfile.rag -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/rag-engine:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/rag-engine:latest

docker build -f docker/Dockerfile.mcp -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/mcp:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/mcp:latest

docker build -f docker/Dockerfile.ui -t asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/streamlit-ui:latest .
docker push asia-southeast1-docker.pkg.dev/<PROJECT_ID>/<REPO>/streamlit-ui:latest
```

### Terraform configuration

Create `infra/terraform/terraform.tfvars` and set:

```hcl
project_id = "your-project-id"
region     = "asia-southeast1"

artifact_repo_name = "rag-platform-repo"

rag_service_name = "rag-engine"
mcp_service_name = "mcp-server"
ui_service_name  = "streamlit-ui"

bucket_name = "your-project-docs-bucket"

documents_prefix   = "documents"
faiss_index_prefix = "indexes/faiss"

rag_container_image = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/rag-engine:latest"
mcp_container_image = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/mcp:latest"
ui_container_image  = "asia-southeast1-docker.pkg.dev/your-project-id/rag-platform-repo/streamlit-ui:latest"

vector_store    = "faiss"
retriever       = "hybrid"
reranker        = "simple"
generator       = "gemini"
document_source = "gcs"

top_k = 3

gemini_api_key = ""

allow_unauthenticated = true
deploy_ui             = true
```

### Apply infrastructure

```bash
cd infra/terraform
terraform init
terraform fmt
terraform plan
terraform apply
```

## Post-deployment validation checklist

After deployment, verify:

- `rag-engine` Cloud Run service is healthy
- `streamlit-ui` opens successfully
- empty GCS bucket field in UI uses the default deployed bucket
- manually entered GCS bucket overrides the default bucket
- local source switching still works
- query and search-only endpoints return expected results
- MCP server can still reach the RAG service

## Destroying cloud infrastructure

To simulate a clean delivery and redeployment cycle:

```bash
cd infra/terraform
terraform destroy
```

Recommended before destroy:

- export or note current Cloud Run service URLs
- confirm whether GCS bucket contents should be preserved or removed
- ensure any manually uploaded test documents are backed up if needed

## Known current scope and limitations

- the UI is lightweight and operational, not a production front-end
- enterprise auth and advanced security hardening are not yet finalized
- current validation focuses on GCP deployment, while cloud portability is demonstrated architecturally and through Terraform-driven patterns rather than live multi-cloud rollout
- evaluation framework is not yet fully completed to final month-3 depth

