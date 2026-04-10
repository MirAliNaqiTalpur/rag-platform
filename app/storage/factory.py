import os

from app.storage.local_store import LocalStorageBackend
from app.storage.gcs_store import GCSStorageBackend


def get_storage_backend(default_local_path: str = "data/documents"):
    source = os.getenv("DOCUMENT_SOURCE", "local").lower()

    if source == "gcs":
        bucket_name = os.getenv("GCS_BUCKET_NAME")
        if not bucket_name:
            raise ValueError("GCS_BUCKET_NAME must be set when DOCUMENT_SOURCE=gcs")
        return GCSStorageBackend(bucket_name=bucket_name)

    return LocalStorageBackend(base_path=default_local_path)