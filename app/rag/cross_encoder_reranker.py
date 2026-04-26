from sentence_transformers import CrossEncoder


class CrossEncoderReranker:
    def __init__(self):
        self.model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
        self.model = CrossEncoder(self.model_name)

    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def rerank(self, query, documents, top_k=None):
        if not documents:
            return []

        final_top_k = top_k or len(documents)

        pairs = [(query, self._doc_text(doc)) for doc in documents]
        scores = self.model.predict(pairs)

        ranked = sorted(
            zip(scores, documents),
            key=lambda item: item[0],
            reverse=True,
        )

        return [doc for score, doc in ranked[:final_top_k]]