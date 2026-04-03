import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class FAISSStore:
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = None
        self.documents = []

    def add_documents(self, docs):
        embeddings = self.embedding_model.encode(docs)
        dim = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings.astype("float32"))

        self.documents = docs

    def search(self, query, top_k=3):
        query_vec = self.embedding_model.encode([query])
        D, I = self.index.search(query_vec.astype("float32"), top_k)

        return [self.documents[i] for i in I[0]]
    
    def save(self, path="data/faiss_index"):
        if not os.path.exists(path):
            os.makedirs(path)

        faiss.write_index(self.index, os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "docs.txt"), "w", encoding="utf-8") as f:
            for doc in self.documents:
                f.write(doc.replace("\n", " ") + "\n")
                
    def load(self, path="data/faiss_index"):
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))

        with open(os.path.join(path, "docs.txt"), "r", encoding="utf-8") as f:
            self.documents = f.readlines()