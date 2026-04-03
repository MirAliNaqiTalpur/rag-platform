from fastapi import FastAPI
from app.rag.engine import RAGEngine

app = FastAPI()
rag = RAGEngine()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/query")
def query(payload: dict):
    query = payload.get("query")
    top_k = payload.get("top_k")
    return rag.query(query=query, top_k=top_k)

@app.post("/search")
def search(payload: dict):
    query = payload.get("query")
    top_k = payload.get("top_k")
    return rag.search_only(query=query, top_k=top_k)