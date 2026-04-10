from abc import ABC, abstractmethod
from typing import List


class StorageBackend(ABC):
    @abstractmethod
    def list_files(self, prefix: str = "") -> List[str]:
        pass

    @abstractmethod
    def read_text(self, path: str) -> str:
        pass

    @abstractmethod
    def read_bytes(self, path: str) -> bytes:
        pass