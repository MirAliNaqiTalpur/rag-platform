import re
from app.vectorstore.factory import get_vector_store
from app.core.config import CONFIG
from app.rag.base_retriever import BaseRetriever
from app.vectorstore.utils import initialize_store


class HybridRetriever(BaseRetriever):
    STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "of", "to", "in", "and", "for", "with", "on"
    }

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

    def _normalize_words(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def keyword_score(self, query, doc):
        query_words = self._normalize_words(query)
        doc_words = self._normalize_words(self._doc_text(doc))
        doc_word_text = " ".join(doc_words)
        return sum(doc_word_text.count(w) for w in query_words)

    def retrieve(self, query, top_k=3):
        dense_results = self.store.search(query, top_k=top_k * 5)

        scored = []
        seen = set()

        for doc in dense_results:
            key = self._doc_key(doc)
            if not key or key in seen:
                continue
            seen.add(key)

            score = self.keyword_score(query, doc)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored[:top_k]]