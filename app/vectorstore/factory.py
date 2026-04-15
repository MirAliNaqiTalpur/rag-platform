from app.vectorstore.faiss_store import FAISSStore
from app.vectorstore.memory_store import InMemoryStore
from app.vectorstore.chroma_store import ChromaStore


def get_vector_store(config):
    provider = config["vector_store"]

    if provider == "faiss":
        return FAISSStore()
    if provider == "memory":
        return InMemoryStore()
    if provider == "chroma":
        return ChromaStore()

    raise ValueError(f"Unsupported vector store: {provider}")