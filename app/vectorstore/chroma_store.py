from __future__ import annotations

from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


class ChromaStore:
    def __init__(self, collection_name: str = "documents") -> None:
        self.collection_name = collection_name
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.client = chromadb.EphemeralClient()
        self.collection = self.client.get_or_create_collection(name=self.collection_name)
        self.documents: list[Any] = []

    def _doc_text(self, doc: Any) -> str:
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def _doc_id(self, doc: Any, idx: int) -> str:
        if isinstance(doc, dict):
            doc_id = doc.get("id")
            if doc_id is not None:
                return str(doc_id)
        return f"doc_{idx}"

    def add_documents(self, docs: list[Any]) -> None:
        self.documents = docs

        try:
            existing = self.collection.get()
            existing_ids = existing.get("ids", [])
            if existing_ids:
                self.collection.delete(ids=existing_ids)
        except Exception:
            pass

        if not docs:
            return

        texts = [self._doc_text(doc) for doc in docs]
        ids = [self._doc_id(doc, i) for i, doc in enumerate(docs)]
        embeddings = self.embedding_model.encode(texts).tolist()

        metadatas = []
        for i, doc in enumerate(docs):
            if isinstance(doc, dict):
                metadata = doc.get("metadata", {}) or {}
                metadata = {k: str(v) for k, v in metadata.items()}
            else:
                metadata = {}
            metadata["source_index"] = str(i)
            metadatas.append(metadata)

        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def search(self, query: str, top_k: int = 3) -> list[Any]:
        query_embedding = self.embedding_model.encode([query]).tolist()

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        matched_docs: list[Any] = []
        metadatas = results.get("metadatas", [[]])[0]

        for metadata in metadatas:
            try:
                idx = int(metadata.get("source_index", -1))
            except Exception:
                idx = -1

            if 0 <= idx < len(self.documents):
                matched_docs.append(self.documents[idx])

        return matched_docs

    def save(self, path: str | None = None) -> None:
        # Ephemeral client does not persist to disk.
        return

    def load(self, path: str | None = None) -> None:
        # Nothing to load for ephemeral mode.
        return