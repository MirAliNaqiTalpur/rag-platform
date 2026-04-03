from app.vectorstore.faiss_store import FAISSStore
from app.vectorstore.memory_store import InMemoryStore


def get_vector_store(config):
    provider = config["vector_store"]

    if provider == "faiss":
        return FAISSStore()

    if provider == "memory":
        return InMemoryStore()

    raise ValueError(f"Unsupported vector store: {provider}")