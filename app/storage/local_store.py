from pathlib import Path
from typing import List

from app.storage.base import StorageBackend


class LocalStorageBackend(StorageBackend):
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)

    def list_files(self, prefix: str = "") -> List[str]:
        root = self.base_path / prefix if prefix else self.base_path
        if not root.exists():
            return []

        files = []
        for path in root.rglob("*"):
            if path.is_file():
                files.append(str(path))
        return files

    def read_text(self, path: str) -> str:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    def read_bytes(self, path: str) -> bytes:
        with open(path, "rb") as f:
            return f.read()