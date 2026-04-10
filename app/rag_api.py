import os
from fastapi import FastAPI
from app.rag.engine import RAGEngine
from app.mcp.schemas import AnswerRequest
from dotenv import load_dotenv
load_dotenv(".env.local")

app = FastAPI()
rag = RAGEngine()


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


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/models")
def get_models():
    return _get_model_config()


@app.post("/query")
def query_docs(request: AnswerRequest):
    return rag.query(
        query=request.query,
        top_k=request.top_k,
        model_name=request.model
    )


@app.post("/search")
def search(payload: dict):
    query = payload.get("query")
    top_k = payload.get("top_k")
    return rag.search_only(query=query, top_k=top_k)