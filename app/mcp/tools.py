from typing import Any, Dict, List

from app.rag.engine import RAGEngine
from app.mcp.schemas import (
    SearchRequest,
    AnswerRequest,
    MetadataSearchRequest,
)


def get_rag_engine():
    return RAGEngine()


def _normalize_documents(docs: List[Any]) -> List[Dict[str, Any]]:
    normalized = []

    for doc in docs:
        if isinstance(doc, dict):
            normalized.append({
                "id": doc.get("id", ""),
                "text": doc.get("text", ""),
                "metadata": doc.get("metadata", {}) or {},
            })
        else:
            normalized.append({
                "id": "",
                "text": str(doc),
                "metadata": {},
            })

    return normalized


def _build_result(
    query: str,
    answer: str = None,
    documents: List[Any] = None,
    latency: Dict[str, float] = None,
) -> Dict[str, Any]:
    return {
        "status": "success",
        "query": query,
        "answer": answer,
        "documents": _normalize_documents(documents or []),
        "latency": latency,
    }


def search_documents(query: str, top_k: int = 3) -> Dict[str, Any]:
    req = SearchRequest(query=query, top_k=top_k)
    result = get_rag_engine().search_only(query=req.query, top_k=req.top_k)

    return _build_result(
        query=req.query,
        documents=result.get("documents", []),
        latency=result.get("latency"),
    )


def answer_query(query: str, top_k: int = 3, model: str = None) -> Dict[str, Any]:
    req = AnswerRequest(query=query, top_k=top_k, model=model)
    result = get_rag_engine().query(
        query=req.query,
        top_k=req.top_k,
        model_name=req.model,
    )

    return _build_result(
        query=req.query,
        answer=result.get("answer"),
        documents=result.get("documents", []),
        latency=result.get("latency"),
    )


def search_by_metadata(
    query: str,
    category: str = None,
    filetype: str = None,
    top_k: int = 3,
) -> Dict[str, Any]:
    req = MetadataSearchRequest(
        query=query,
        category=category,
        filetype=filetype,
        top_k=top_k,
    )

    result = get_rag_engine().search_only(query=req.query, top_k=req.top_k)
    docs = result.get("documents", [])

    filtered_docs = []
    for doc in docs:
        if not isinstance(doc, dict):
            continue

        metadata = doc.get("metadata", {}) or {}

        if req.category and metadata.get("category") != req.category:
            continue

        if req.filetype and metadata.get("filetype") != req.filetype:
            continue

        filtered_docs.append(doc)

    return _build_result(
        query=req.query,
        documents=filtered_docs,
        latency=result.get("latency"),
    )


def dispatch_tool(method: str, params: dict) -> Dict[str, Any]:
    if method == "search_documents":
        req = SearchRequest(**params)
        return search_documents(query=req.query, top_k=req.top_k)

    if method == "answer_query":
        req = AnswerRequest(**params)
        return answer_query(query=req.query, top_k=req.top_k, model=req.model)

    if method == "search_by_metadata":
        req = MetadataSearchRequest(**params)
        return search_by_metadata(
            query=req.query,
            category=req.category,
            filetype=req.filetype,
            top_k=req.top_k,
        )

    raise ValueError(f"Unknown method: {method}")