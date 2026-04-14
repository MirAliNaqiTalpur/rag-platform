import os
import shutil
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.core.config import CONFIG
from app.ingestion.loader import load_documents
from app.mcp.schemas import AnswerRequest
from app.rag.engine import RAGEngine
from app.vectorstore.factory import get_vector_store
from app.rag.gemini_generator import ModelBusyError, UpstreamModelError

load_dotenv(".env.local")

app = FastAPI()

LOCAL_DOCUMENTS_DIR = "data/documents"
FAISS_INDEX_DIR = "data/faiss_index"


@lru_cache(maxsize=1)
def get_rag() -> RAGEngine:
    return RAGEngine()


def _get_model_config() -> dict[str, Any]:
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
    unique_models: list[str] = []
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


def _get_nonempty_string(payload: dict, key: str, default: str = "") -> str:
    value = payload.get(key)
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip()
    return str(value)


def _update_runtime_config(payload: dict) -> dict[str, str | int]:
    document_source = _get_nonempty_string(
        payload, "document_source", os.getenv("DOCUMENT_SOURCE", "local")
    ).lower()

    vector_store = _get_nonempty_string(
        payload, "vector_store", os.getenv("VECTOR_STORE", "faiss")
    )
    retriever = _get_nonempty_string(
        payload, "retriever", os.getenv("RETRIEVER", "hybrid")
    )
    reranker = _get_nonempty_string(
        payload, "reranker", os.getenv("RERANKER", "simple")
    )
    generator = _get_nonempty_string(
        payload, "generator", os.getenv("GENERATOR", "gemini")
    )

    top_k_raw = payload.get("top_k")
    top_k = int(top_k_raw) if top_k_raw not in (None, "") else int(os.getenv("TOP_K", "3"))

    default_model = _get_nonempty_string(
        payload,
        "default_model",
        os.getenv("DEFAULT_MODEL", "gemini-3.1-flash-lite-preview"),
    )
    available_models = _get_nonempty_string(
        payload,
        "available_models",
        os.getenv("AVAILABLE_MODELS", default_model),
    )

    os.environ["DOCUMENT_SOURCE"] = document_source
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

    return {
        "document_source": document_source,
        "top_k": top_k,
    }


def _set_gcs_runtime_config(bucket_name: str, prefix: str) -> None:
    os.environ["GCS_BUCKET_NAME"] = bucket_name
    os.environ["GCS_PREFIX"] = prefix


def _clear_gcs_runtime_config() -> None:
    os.environ.pop("GCS_BUCKET_NAME", None)
    os.environ.pop("GCS_PREFIX", None)


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
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, str]:
    cache_info = get_rag.cache_info()
    if cache_info.currsize > 0:
        return {"status": "ready"}
    raise HTTPException(
        status_code=503,
        detail="RAG engine is not warmed yet. Call /warmup or reload the dataset first."
    )


@app.get("/models")
def get_models() -> dict[str, Any]:
    return _get_model_config()


@app.post("/warmup")
def warmup() -> dict[str, str]:
    try:
        get_rag()
        return {"status": "ok", "message": "RAG engine initialized successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Warmup failed: {e}") from e


@app.post("/reload-dataset")
def reload_dataset(payload: dict) -> dict[str, Any]:
    try:
        deployed_default_bucket = os.getenv("GCS_BUCKET_NAME", "").strip()
        deployed_default_prefix = os.getenv("GCS_PREFIX", "").strip()

        runtime = _update_runtime_config(payload)
        document_source = runtime["document_source"]

        if document_source == "gcs":
            gcs_bucket_name = _get_nonempty_string(payload, "gcs_bucket_name", "")
            gcs_prefix = _get_nonempty_string(payload, "gcs_prefix", "")

            if gcs_bucket_name:
                resolved_bucket = gcs_bucket_name
                resolved_prefix = gcs_prefix
                bucket_message = f"Using GCS bucket from UI: '{resolved_bucket}'."
            else:
                resolved_bucket = deployed_default_bucket
                resolved_prefix = deployed_default_prefix

                if not resolved_bucket:
                    _clear_gcs_runtime_config()
                    get_rag.cache_clear()
                    return {
                        "status": "success",
                        "document_source": "gcs",
                        "gcs_bucket_name": "",
                        "gcs_prefix": "",
                        "total_documents_loaded": 0,
                        "message": "GCS source selected, but no bucket was provided and no default deployed bucket is configured.",
                    }

                bucket_message = f"Using default deployed GCS bucket: '{resolved_bucket}'."

            _set_gcs_runtime_config(resolved_bucket, resolved_prefix)

        else:
            _clear_gcs_runtime_config()
            bucket_message = ""

        total_docs = _rebuild_vector_store()
        get_rag.cache_clear()
        get_rag()

        message = (
            f"{bucket_message} Dataset loaded successfully."
            if total_docs > 0
            else f"{bucket_message} No documents found in selected source."
        ).strip()

        return {
            "status": "success",
            "document_source": os.getenv("DOCUMENT_SOURCE", "local"),
            "gcs_bucket_name": os.getenv("GCS_BUCKET_NAME", ""),
            "gcs_prefix": os.getenv("GCS_PREFIX", ""),
            "total_documents_loaded": total_docs,
            "message": message,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dataset reload failed: {e}") from e


@app.post("/query")
def query_docs(request: AnswerRequest):
    rag = get_rag()

    try:
        return rag.query(
            query=request.query,
            top_k=request.top_k,
            model_name=request.model,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    except ModelBusyError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e

    except UpstreamModelError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Internal query error: {e}"
        ) from e


@app.post("/search")
def search(payload: dict):
    rag = get_rag()
    query = payload.get("query")
    top_k = payload.get("top_k")
    return rag.search_only(query=query, top_k=top_k)