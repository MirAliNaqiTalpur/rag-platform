import os


def get_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or value.strip() == "":
        return default
    return int(value)


CONFIG = {
    "vector_store": os.getenv("VECTOR_STORE", "faiss"),
    "retriever": os.getenv("RETRIEVER", "hybrid"),
    "reranker": os.getenv("RERANKER", "simple"),
    "top_k": get_int_env("TOP_K", 3),
    "generator": os.getenv("GENERATOR", "gemini"),

    # Index persistence settings
    "index_storage": os.getenv("INDEX_STORAGE", "local").lower(),
    "gcs_index_bucket": os.getenv("GCS_INDEX_BUCKET", os.getenv("GCS_BUCKET_NAME", "")).strip(),
    "gcs_index_prefix": os.getenv("GCS_INDEX_PREFIX", "indexes/faiss").strip(),
}