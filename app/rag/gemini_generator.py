import os
from google import genai
from app.rag.base_generator import BaseGenerator

class GeminiGenerator(BaseGenerator):
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")

        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

        self.client = genai.Client(api_key=api_key)

    def generate(self, query, documents):
        context = "\n".join(documents)

        prompt = f"""
Answer the question based only on the context below.

Context:
{context}

Question:
{query}

Answer:
"""

        response = self.client.models.generate_content(
            model="gemini-3.1-flash-lite-preview",
            contents=prompt
        )

        return response.text