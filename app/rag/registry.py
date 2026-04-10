from app.rag.hybrid_retriever import HybridRetriever
from app.rag.retriever import DenseRetriever
from app.rag.simple_retriever import SimpleRetriever
from app.rag.metadata_retriever import MetadataAwareRetriever

class RetrieverRegistry:
    def __init__(self):
        self._registry = {
            "dense": DenseRetriever,
            "simple": SimpleRetriever,
            "hybrid": HybridRetriever,
            "metadata": MetadataAwareRetriever
        }

    def get(self, name):
        retriever_class = self._registry.get(name)

        if retriever_class is None:
            raise ValueError(f"Unknown retriever strategy: {name}")

        return retriever_class()