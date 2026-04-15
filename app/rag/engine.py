import time
from app.rag.registry import RetrieverRegistry
from app.rag.reranker_registry import RerankerRegistry
from app.rag.generator_registry import GeneratorRegistry
from app.core.config import CONFIG


def to_ms(seconds: float) -> float:
    """Convert seconds to milliseconds with sensible precision."""
    return round(seconds * 1000, 2)


class RAGEngine:
    def __init__(self):
        retriever_registry = RetrieverRegistry()
        reranker_registry = RerankerRegistry()
        generator_registry = GeneratorRegistry()

        self.retriever = retriever_registry.get(CONFIG["retriever"])
        self.reranker = reranker_registry.get(CONFIG["reranker"])
        self.generator = generator_registry.get(CONFIG["generator"])

    def query(
        self,
        user_query: str = None,
        query: str = None,
        top_k: int = None,
        model_name: str = None,
    ):
        total_start = time.perf_counter()

        final_query = query if query is not None else user_query
        final_top_k = top_k if top_k is not None else CONFIG["top_k"]

        retrieval_start = time.perf_counter()
        documents = self.retriever.retrieve(final_query, top_k=final_top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        rerank_start = time.perf_counter()
        if self.reranker:
            reranked_documents = self.reranker.rerank(final_query, documents)
        else:
            reranked_documents = documents
        rerank_time = time.perf_counter() - rerank_start

        generation_start = time.perf_counter()
        if self.generator:
            answer = self.generator.generate(
                final_query,
                reranked_documents,
                model_name=model_name,
            )
        else:
            answer = "No generator configured"
        generation_time = time.perf_counter() - generation_start

        total_time = time.perf_counter() - total_start

        return {
            "query": final_query,
            "documents": reranked_documents,
            "answer": answer,
            "latency": {
                "retrieval_ms": to_ms(retrieval_time),
                "reranking_ms": to_ms(rerank_time),
                "generation_ms": to_ms(generation_time),
                "total_ms": to_ms(total_time),
            },
        }

    def search_only(self, query: str, top_k: int = None):
        final_top_k = top_k if top_k is not None else CONFIG["top_k"]

        retrieval_start = time.perf_counter()
        documents = self.retriever.retrieve(query, top_k=final_top_k)
        retrieval_time = time.perf_counter() - retrieval_start

        rerank_start = time.perf_counter()
        if self.reranker:
            reranked_documents = self.reranker.rerank(query, documents)
        else:
            reranked_documents = documents
        rerank_time = time.perf_counter() - rerank_start

        total_time = retrieval_time + rerank_time

        return {
            "query": query,
            "documents": reranked_documents,
            "latency": {
                "retrieval_ms": to_ms(retrieval_time),
                "reranking_ms": to_ms(rerank_time),
                "total_ms": to_ms(total_time),
            },
        }