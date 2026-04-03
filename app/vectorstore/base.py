from abc import ABC, abstractmethod


class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, docs):
        pass

    @abstractmethod
    def search(self, query, top_k=3):
        pass