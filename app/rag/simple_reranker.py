from app.rag.base_reranker import BaseReranker

class SimpleReranker(BaseReranker):
    def rerank(self, query, documents):
        query_words = query.lower().split()

        scored_docs = []
        for doc in documents:
            doc_text = doc.lower()
            score = sum(1 for word in query_words if word in doc_text)
            scored_docs.append((score, doc))

        scored_docs.sort(key=lambda x: x[0], reverse=True)

        return [doc for score, doc in scored_docs]