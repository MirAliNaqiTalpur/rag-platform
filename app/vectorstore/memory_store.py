from sentence_transformers import SentenceTransformer
from app.vectorstore.base import VectorStore


class InMemoryStore(VectorStore):
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.documents = []
        self.embeddings = []

    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def add_documents(self, docs):
        self.documents = docs
        texts = [self._doc_text(doc) for doc in docs]
        self.embeddings = self.embedding_model.encode(texts).tolist()

    def search(self, query, top_k=3):
        query_vec = self.embedding_model.encode([query])[0]

        scored = []
        for doc, emb in zip(self.documents, self.embeddings):
            score = self.cosine_similarity(query_vec, emb)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]

    def save(self, path="data/memory_store"):
        pass

    def load(self, path="data/memory_store"):
        pass

    def cosine_similarity(self, a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)