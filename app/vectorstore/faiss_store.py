import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class FAISSStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.documents = []

    def _doc_text(self, doc):
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def add_documents(self, docs):
        texts = [self._doc_text(doc) for doc in docs]
        embeddings = self.embedding_model.encode(texts)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.array(embeddings, dtype="float32"))

        self.documents = docs

    def search(self, query, top_k=3):
        query_vec = self.embedding_model.encode([query])
        _, indices = self.index.search(np.array(query_vec, dtype="float32"), top_k)

        results = []
        for i in indices[0]:
            if 0 <= i < len(self.documents):
                results.append(self.documents[i])
        return results

    def save(self, path="data/faiss_index"):
        if not os.path.exists(path):
            os.makedirs(path)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "docs.json"), "w", encoding="utf-8") as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)

    def load(self, path="data/faiss_index"):
        index_path = os.path.join(path, "index.faiss")
        docs_path = os.path.join(path, "docs.json")

        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            raise FileNotFoundError(f"FAISS index files not found in: {path}")

        self.index = faiss.read_index(index_path)

        with open(docs_path, "r", encoding="utf-8") as f:
            self.documents = json.load(f)