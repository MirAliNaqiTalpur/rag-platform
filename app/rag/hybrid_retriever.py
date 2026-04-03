from app.vectorstore.factory import get_vector_store
from app.core.config import CONFIG
from app.rag.base_retriever import BaseRetriever
from app.vectorstore.utils import initialize_store


class HybridRetriever(BaseRetriever):
    def __init__(self):
        self.store = get_vector_store(CONFIG)
        self.store = initialize_store(self.store)

    def keyword_score(self, query, doc):
        query_words = query.lower().split()
        doc_text = doc.lower()
        return sum(doc_text.count(w) for w in query_words)

    def retrieve(self, query, top_k=3):
        dense_results = self.store.search(query, top_k=top_k * 5)

        scored = []
        for doc in dense_results:
            score = self.keyword_score(query, doc)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        final_docs = [doc for score, doc in scored[:top_k]]
        return final_docs