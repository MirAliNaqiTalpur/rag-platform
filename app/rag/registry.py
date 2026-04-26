from app.rag.hybrid_retriever import HybridRetriever
from app.rag.retriever import DenseRetriever
from app.rag.simple_retriever import SimpleRetriever
from app.rag.metadata_retriever import MetadataAwareRetriever
from app.rag.bm25_retriever import BM25Retriever
from app.rag.cross_encoder_reranker import CrossEncoderReranker


class RetrieverRegistry:
    def __init__(self):
        self._registry = {
            "dense": DenseRetriever,
            "simple": SimpleRetriever,
            "hybrid": HybridRetriever,
            "metadata": MetadataAwareRetriever,
            "bm25": BM25Retriever,
        }

    def get(self, name):
        retriever_class = self._registry.get(name)
        if retriever_class is None:
            raise ValueError(f"Unknown retriever strategy: {name}")
        return retriever_class()