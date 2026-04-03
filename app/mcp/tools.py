from functools import lru_cache
from app.rag.engine import RAGEngine

@lru_cache(maxsize=1)
def get_rag_engine():
    return RAGEngine()

def search_documents(query: str, top_k: int = 3):
    return get_rag_engine().search_only(query=query, top_k=top_k)

def answer_query(query: str, top_k: int = 3):
    return get_rag_engine().query(query=query, top_k=top_k)

def dispatch_tool(method: str, params: dict):
    query = params.get("query")
    top_k = params.get("top_k", 3)

    if not query:
        raise ValueError("Missing required parameter: query")

    if method == "search_documents":
        return search_documents(query=query, top_k=top_k)

    if method == "answer_query":
        return answer_query(query=query, top_k=top_k)

    raise ValueError(f"Unknown method: {method}")