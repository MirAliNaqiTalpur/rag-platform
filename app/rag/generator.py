from app.rag.base_generator import BaseGenerator

class MockGenerator(BaseGenerator):
    def generate(self, query, documents):
        context = "\n".join(documents)

        response = f"""
Answer the question based on the context below.

Context:
{context}

Question:
{query}

Answer:
"""
        return response.strip()