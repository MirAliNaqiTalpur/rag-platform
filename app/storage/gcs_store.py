from typing import List
from google.cloud import storage

from app.storage.base import StorageBackend


class GCSStorageBackend(StorageBackend):
    def __init__(self, bucket_name: str):
        self.bucket_name = bucket_name
        self.client = storage.Client()

    def list_files(self, prefix: str = "") -> List[str]:
        blobs = self.client.list_blobs(self.bucket_name, prefix=prefix)
        return [blob.name for blob in blobs if not blob.name.endswith("/")]

    def read_text(self, path: str) -> str:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(path)
        return blob.download_as_text(encoding="utf-8")

    def read_bytes(self, path: str) -> bytes:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(path)
        return blob.download_as_bytes()