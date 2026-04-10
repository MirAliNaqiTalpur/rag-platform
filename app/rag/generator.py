from app.rag.base_generator import BaseGenerator


class MockGenerator(BaseGenerator):
    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def generate(self, query, documents, model_name=None):
        context = "\n\n".join(self._doc_text(doc) for doc in documents)

        return f"""Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:"""