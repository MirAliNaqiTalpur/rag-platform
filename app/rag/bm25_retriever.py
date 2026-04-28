"""
BM25Retriever — sparse lexical retrieval baseline.

Implements pure BM25 retrieval over the full document corpus, independent
of any dense embedding store. This is the standard sparse-retrieval
baseline cited across IR literature (Robertson & Zaragoza, 2009).

Why this matters academically:
  - The previous implementation called `self.store.search(query, top_k=1000)`
    to gather "candidates" before running BM25 over them. This made BM25
    a *re-ranker over dense candidates*, not a true sparse retriever:
      • BM25's lexical-match advantage was filtered out by dense ranking
      • On large corpora (real FiQA = 57K docs), 1000 candidates ≪ corpus,
        so BM25 was scoring a heavily biased subset
      • The implementation rebuilt the BM25 index per query, an O(N) cost
        each time
  - This implementation is corpus-independent: it loads documents once
    via the platform's `load_documents()` and builds a single BM25 index
    that is queried for every search.
  - The result: a clean ablation baseline. Differences between BM25,
    dense, and hybrid now reflect retriever quality, not implementation
    artifacts.

Reference:
  Robertson, S., & Zaragoza, H. (2009). The Probabilistic Relevance
  Framework: BM25 and Beyond. Foundations and Trends in IR.

Registered via:  RETRIEVER=bm25
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

from rank_bm25 import BM25Okapi

from app.rag.base_retriever import BaseRetriever

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Tokenizer — deterministic, reproducible, matches HybridRetriever
# ─────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Deterministic tokenizer for BM25.

    - Lowercase alphanumeric tokens of length >= 2
    - No stemming, no stopword removal (keeps the BM25 baseline pure)
    - Identical to HybridRetriever's tokenizer for fair comparison
    """
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) >= 2]


# ─────────────────────────────────────────────────────────────────────
# Module-level cache for the BM25 index
# ─────────────────────────────────────────────────────────────────────
# Build once per dataset reload, query repeatedly. Mirrors the platform's
# pattern of caching expensive resources (e.g., the FAISS embedding model).

_bm25_index_cache: Dict[str, Any] = {
    "bm25": None,
    "documents": None,
    "tokenized_corpus": None,
    "doc_count": 0,
}


def _build_index(documents_dir: str = "data/documents") -> None:
    """Build the BM25 index from the platform's document folder.

    Uses the same `load_documents()` that `/reload-dataset` calls, so
    BM25 sees the exact same corpus as the dense retriever — guaranteed
    fair comparison in ablation.
    """
    # Lazy import keeps `import app.rag.bm25_retriever` cheap and lets
    # unit tests run without the full ingestion stack.
    from app.ingestion.loader import load_documents

    docs = load_documents(documents_dir)

    if not docs:
        logger.warning("BM25Retriever: no documents loaded; index will be empty.")
        _bm25_index_cache["bm25"] = None
        _bm25_index_cache["documents"] = []
        _bm25_index_cache["tokenized_corpus"] = []
        _bm25_index_cache["doc_count"] = 0
        return

    tokenized = [_tokenize(doc.get("text", "")) for doc in docs]

    # Drop documents with empty tokenization (would break BM25 normalization)
    non_empty = [(d, t) for d, t in zip(docs, tokenized) if t]
    if len(non_empty) != len(docs):
        dropped = len(docs) - len(non_empty)
        logger.warning(f"BM25Retriever: dropped {dropped} document(s) with empty tokenization.")

    if not non_empty:
        _bm25_index_cache["bm25"] = None
        _bm25_index_cache["documents"] = []
        _bm25_index_cache["tokenized_corpus"] = []
        _bm25_index_cache["doc_count"] = 0
        return

    documents = [d for d, _ in non_empty]
    tokenized_corpus = [t for _, t in non_empty]

    _bm25_index_cache["documents"] = documents
    _bm25_index_cache["tokenized_corpus"] = tokenized_corpus
    _bm25_index_cache["bm25"] = BM25Okapi(tokenized_corpus, k1=1.5, b=0.75)
    _bm25_index_cache["doc_count"] = len(documents)

    logger.info(f"BM25Retriever: indexed {len(documents)} document(s).")


def reset_bm25_index() -> None:
    """Clear the cached BM25 index. Call when the dataset reloads."""
    _bm25_index_cache["bm25"] = None
    _bm25_index_cache["documents"] = None
    _bm25_index_cache["tokenized_corpus"] = None
    _bm25_index_cache["doc_count"] = 0
    logger.info("BM25Retriever: index cache cleared.")


# ─────────────────────────────────────────────────────────────────────
# BM25 Retriever
# ─────────────────────────────────────────────────────────────────────

class BM25Retriever(BaseRetriever):
    """Sparse BM25 retriever, independent of the dense vector store.

    The index is built lazily on first retrieve() call from the same
    `data/documents/` corpus the dense retriever uses, then cached
    until reset_bm25_index() is called (e.g., on dataset reload).

    Returned documents conform to the platform schema:
        {"id": str, "text": str, "metadata": {...}, "score": float}
    where "score" is the BM25 relevance score.
    """

    K1 = 1.5
    B = 0.75

    def __init__(self):
        """Zero-arg constructor — matches RetrieverRegistry().get(name) convention."""
        # Index is built lazily on first retrieve() call; nothing to do here.
        pass

    def _doc_key(self, doc: Any) -> str:
        """Deduplication key — mirrors SimpleRetriever pattern."""
        if isinstance(doc, dict):
            return doc.get("id") or doc.get("text", "").strip()
        return str(doc).strip()

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the top_k documents ranked by BM25 score.

        Args:
            query: User query string.
            top_k: Number of documents to return (must be >= 1).

        Returns:
            List of document dicts (length <= top_k), ordered by descending
            BM25 score with deterministic tiebreaking. Returns [] if the
            corpus is empty or the query tokenizes to nothing.
        """
        # Lazy index build on first call
        if _bm25_index_cache["bm25"] is None:
            _build_index()

        bm25 = _bm25_index_cache["bm25"]
        documents = _bm25_index_cache["documents"]

        if bm25 is None or not documents:
            logger.warning("BM25Retriever.retrieve: empty index; returning [].")
            return []

        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        query_tokens = _tokenize(query)
        if not query_tokens:
            logger.debug(f"BM25Retriever: query tokenized to empty list: {query!r}")
            return []

        scores = bm25.get_scores(query_tokens)

        # Rank by score descending; deterministic tiebreak by index
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i),
        )

        # Apply dedup before truncating to top_k
        results: List[Dict[str, Any]] = []
        seen = set()
        for idx in ranked_indices:
            if len(results) >= top_k:
                break
            doc = dict(documents[idx])  # shallow copy — don't mutate cache
            key = self._doc_key(doc)
            if not key or key in seen:
                continue
            seen.add(key)
            doc["score"] = float(scores[idx])
            results.append(doc)

        return results
