from app.vectorstore.factory import get_vector_store
from app.core.config import CONFIG
from app.rag.base_retriever import BaseRetriever
from app.vectorstore.utils import initialize_store


class DenseRetriever(BaseRetriever):
    def __init__(self):
        self.store = get_vector_store(CONFIG)
        self.store = initialize_store(self.store)

    def _doc_key(self, doc):
        if isinstance(doc, dict):
            return doc.get("id") or doc.get("text", "").strip()
        return str(doc).strip()

    def retrieve(self, query, top_k=3):
        results = self.store.search(query, top_k=top_k * 2)

        unique = []
        seen = set()

        for doc in results:
            key = self._doc_key(doc)
            if key and key not in seen:
                unique.append(doc)
                seen.add(key)

        return unique[:top_k]