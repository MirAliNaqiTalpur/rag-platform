import os

CONFIG = {
    "vector_store": os.getenv("VECTOR_STORE", "faiss"),
    "retriever": os.getenv("RETRIEVER", "hybrid"),
    "reranker": os.getenv("RERANKER", "simple"),
    "top_k": int(os.getenv("TOP_K", 3)),
    "generator": os.getenv("GENERATOR", "mock"),
}