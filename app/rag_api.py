import os
import shutil
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from app.core.config import CONFIG
from app.rag.engine import RAGEngine
from app.mcp.schemas import AnswerRequest
from app.ingestion.loader import load_documents
from app.vectorstore.factory import get_vector_store

load_dotenv(".env.local")

app = FastAPI()

LOCAL_DOCUMENTS_DIR = "data/documents"
FAISS_INDEX_DIR = "data/faiss_index"


@lru_cache(maxsize=1)
def get_rag() -> RAGEngine:
    return RAGEngine()


def _get_model_config():
    raw_models = os.getenv("AVAILABLE_MODELS", "").strip()

    if not raw_models:
        raw_models = os.getenv("AVAILABLE_GEMINI_MODELS", "").strip()

    if not raw_models:
        raw_models = "gemini-3.1-flash-lite-preview"

    available_models = [m.strip() for m in raw_models.split(",") if m.strip()]

    default_model = os.getenv("DEFAULT_MODEL", "").strip()
    if not default_model:
        default_model = os.getenv("DEFAULT_GEMINI_MODEL", "").strip()

    if not default_model:
        default_model = available_models[0]

    if default_model not in available_models:
        available_models.insert(0, default_model)

    seen = set()
    unique_models = []
    for model in available_models:
        if model not in seen:
            seen.add(model)
            unique_models.append(model)

    return {
        "default_model": default_model,
        "available_models": unique_models,
        "llm_provider": os.getenv("LLM_PROVIDER", "gemini"),
        "allow_custom_models": os.getenv("ALLOW_CUSTOM_MODELS", "false").strip().lower() == "true",
    }


def _update_runtime_config(payload: dict) -> None:
    document_source = payload.get("document_source", os.getenv("DOCUMENT_SOURCE", "local"))
    gcs_bucket_name = payload.get("gcs_bucket_name", os.getenv("GCS_BUCKET_NAME", ""))
    gcs_prefix = payload.get("gcs_prefix", os.getenv("GCS_PREFIX", ""))

    vector_store = payload.get("vector_store", os.getenv("VECTOR_STORE", "faiss"))
    retriever = payload.get("retriever", os.getenv("RETRIEVER", "hybrid"))
    reranker = payload.get("reranker", os.getenv("RERANKER", "simple"))
    generator = payload.get("generator", os.getenv("GENERATOR", "gemini"))
    top_k = int(payload.get("top_k", os.getenv("TOP_K", 3)))

    default_model = payload.get("default_model", os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite-preview"))
    available_models = payload.get("available_models", os.getenv("AVAILABLE_MODELS", default_model))

    os.environ["DOCUMENT_SOURCE"] = document_source
    os.environ["GCS_BUCKET_NAME"] = gcs_bucket_name
    os.environ["GCS_PREFIX"] = gcs_prefix

    os.environ["VECTOR_STORE"] = vector_store
    os.environ["RETRIEVER"] = retriever
    os.environ["RERANKER"] = reranker
    os.environ["GENERATOR"] = generator
    os.environ["TOP_K"] = str(top_k)

    os.environ["DEFAULT_MODEL"] = default_model
    os.environ["AVAILABLE_MODELS"] = available_models

    CONFIG["vector_store"] = vector_store
    CONFIG["retriever"] = retriever
    CONFIG["reranker"] = reranker
    CONFIG["generator"] = generator
    CONFIG["top_k"] = top_k


def _rebuild_vector_store() -> int:
    if os.path.exists(FAISS_INDEX_DIR):
        shutil.rmtree(FAISS_INDEX_DIR)

    docs = load_documents(LOCAL_DOCUMENTS_DIR)

    if not docs:
        return 0

    store = get_vector_store(CONFIG)
    store.add_documents(docs)

    try:
        store.save()
    except Exception:
        pass

    return len(docs)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return _get_model_config()


@app.post("/reload-dataset")
def reload_dataset(payload: dict):
    try:
        _update_runtime_config(payload)
        total_docs = _rebuild_vector_store()

        # Clear cached engine so next request uses the new source/index/config
        get_rag.cache_clear()

        return {
            "status": "success",
            "document_source": os.getenv("DOCUMENT_SOURCE", "local"),
            "gcs_bucket_name": os.getenv("GCS_BUCKET_NAME", ""),
            "gcs_prefix": os.getenv("GCS_PREFIX", ""),
            "total_documents_loaded": total_docs,
            "message": "Dataset loaded successfully." if total_docs > 0 else "No documents found in selected source.",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset reload failed: {e}")


@app.post("/query")
def query_docs(request: AnswerRequest):
    rag = get_rag()
    return rag.query(
        query=request.query,
        top_k=request.top_k,
        model_name=request.model,
    )


@app.post("/search")
def search(payload: dict):
    rag = get_rag()
    query = payload.get("query")
    top_k = payload.get("top_k")
    return rag.search_only(query=query, top_k=top_k)