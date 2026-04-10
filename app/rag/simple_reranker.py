import re
from app.rag.base_reranker import BaseReranker


class SimpleReranker(BaseReranker):
    STOPWORDS = {
        "what", "is", "are", "the", "a", "an", "of", "to", "in", "and", "for", "with", "on"
    }
    
    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def _normalize_words(self, text):
        words = re.findall(r"\b\w+\b", text.lower())
        return [w for w in words if w not in self.STOPWORDS]

    def rerank(self, query, documents):
        query_words = self._normalize_words(query)

        scored_docs = []
        for doc in documents:
            doc_words = self._normalize_words(self._doc_text(doc))
            doc_word_text = " ".join(doc_words)
            score = sum(doc_word_text.count(word) for word in query_words)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored_docs]