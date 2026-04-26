import re
from rank_bm25 import BM25Okapi

from app.core.config import CONFIG
from app.rag.base_retriever import BaseRetriever
from app.vectorstore.factory import get_vector_store
from app.vectorstore.utils import initialize_store


class BM25Retriever(BaseRetriever):
    STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "of", "to", "in",
        "and", "for", "with", "on", "by", "how", "does", "do"
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

    def _tokenize(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def retrieve(self, query, top_k=3):
        # Use vector store only to access indexed documents.
        candidate_docs = self.store.search(query, top_k=1000)

        unique_docs = []
        seen = set()

        for doc in candidate_docs:
            key = self._doc_key(doc)
            if key and key not in seen:
                unique_docs.append(doc)
                seen.add(key)

        if not unique_docs:
            return []

        tokenized_corpus = [self._tokenize(self._doc_text(doc)) for doc in unique_docs]
        bm25 = BM25Okapi(tokenized_corpus)

        query_tokens = self._tokenize(query)
        scores = bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(scores, unique_docs),
            key=lambda item: item[0],
            reverse=True,
        )

        return [doc for score, doc in ranked[:top_k]]