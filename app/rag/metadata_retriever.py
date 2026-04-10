from app.vectorstore.factory import get_vector_store
from app.core.config import CONFIG
from app.rag.base_retriever import BaseRetriever
from app.vectorstore.utils import initialize_store


class MetadataAwareRetriever(BaseRetriever):
    def __init__(self):
        self.store = get_vector_store(CONFIG)
        self.store = initialize_store(self.store)

    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def _doc_key(self, doc):
        if isinstance(doc, dict):
            return doc.get("id") or doc.get("text", "").strip()
        return str(doc).strip()

    def _match_category(self, query, doc):
        if not isinstance(doc, dict):
            return False

        category = doc.get("metadata", {}).get("category", "").lower()
        query_lower = query.lower()

        return category and category in query_lower

    def retrieve(self, query, top_k=3):
        results = self.store.search(query, top_k=top_k * 5)

        prioritized = []
        regular = []
        seen = set()

        for doc in results:
            key = self._doc_key(doc)
            if not key or key in seen:
                continue
            seen.add(key)

            if self._match_category(query, doc):
                prioritized.append(doc)
            else:
                regular.append(doc)

        combined = prioritized + regular
        return combined[:top_k]