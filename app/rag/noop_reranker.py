from app.rag.base_reranker import BaseReranker

class NoOpReranker(BaseReranker):
    def rerank(self, query, documents):
        return documents