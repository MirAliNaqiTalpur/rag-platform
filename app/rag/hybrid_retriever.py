"""
HybridRetriever — Reciprocal Rank Fusion of dense and sparse retrieval.

Combines dense semantic retrieval (FAISS over sentence embeddings) with
sparse lexical retrieval (BM25 over tokenized text) via Reciprocal Rank
Fusion (RRF), the standard academic approach for hybrid retrieval.

Why this matters academically:
  - The previous implementation labeled "hybrid" was actually
    dense-then-keyword-rerank: it used dense retrieval to produce candidates,
    then sorted them by raw keyword overlap. The dense scores were discarded.
    This is not hybrid retrieval as defined in the IR literature.
  - True hybrid retrieval *fuses* signals from independent retrieval
    paradigms (dense and sparse), exploiting the strengths of each:
    dense captures paraphrase and semantic similarity; sparse captures
    exact-term matches and rare-term importance.
  - Reciprocal Rank Fusion (RRF) is the de-facto standard fusion method
    because it is parameter-free (other than the constant k=60), robust
    across domains, and does not require score calibration between the
    two retrievers.

Fusion formula (Cormack, Clarke & Buettcher 2009):

    RRF(d) = sum over retrievers r of  1 / (k + rank_r(d))

where rank_r(d) is the 1-indexed position of document d in the ranked
list returned by retriever r, k=60 is a smoothing constant, and documents
not present in a retriever's list contribute 0 from that retriever.

Reference:
  Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
  Reciprocal Rank Fusion outperforms Condorcet and individual Rank
  Learning Methods. SIGIR 2009.

Registered via:  RETRIEVER=hybrid
"""

from __future__ import annotations

import logging
import re
from typing import List, Dict, Any

from app.core.config import CONFIG
from app.rag.base_retriever import BaseRetriever
from app.vectorstore.factory import get_vector_store
from app.vectorstore.utils import initialize_store

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────
# RRF smoothing constant
# ─────────────────────────────────────────────────────────────────────
# k=60 is the value originally used by Cormack et al. and now standard
# across the literature. It mildly de-emphasizes top-ranked docs from
# any single retriever, preventing one retriever from dominating fusion.
RRF_K = 60

# How many candidates to ask each retriever for. We retrieve more than
# top_k from each so that fusion has a meaningful candidate pool. 50 is
# a reasonable default for both speed and recall.
CANDIDATE_POOL_SIZE = 50


# ─────────────────────────────────────────────────────────────────────
# Tokenizer for BM25 (matches BM25Retriever for consistency)
# ─────────────────────────────────────────────────────────────────────
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Deterministic tokenizer for BM25; matches BM25Retriever exactly."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text) if len(tok) >= 2]


# ─────────────────────────────────────────────────────────────────────
# Module-level cache for the BM25 index
# ─────────────────────────────────────────────────────────────────────
# We cache the tokenized corpus and BM25 index so they are built once
# per dataset reload, not once per query. This avoids the N-queries × 
# corpus-tokenization cost.

_bm25_cache: Dict[str, Any] = {
    "bm25": None,
    "documents": None,
    "tokenized_corpus": None,
}


def _build_bm25_from_documents(documents: List[Dict[str, Any]]) -> None:
    """Build a BM25 index over the given document list and cache it."""
    from rank_bm25 import BM25Okapi

    tokenized = [_tokenize(d.get("text", "")) for d in documents]
    non_empty = [(d, t) for d, t in zip(documents, tokenized) if t]

    if not non_empty:
        _bm25_cache["bm25"] = None
        _bm25_cache["documents"] = []
        _bm25_cache["tokenized_corpus"] = []
        return

    docs = [d for d, _ in non_empty]
    tokens = [t for _, t in non_empty]

    _bm25_cache["bm25"] = BM25Okapi(tokens, k1=1.5, b=0.75)
    _bm25_cache["documents"] = docs
    _bm25_cache["tokenized_corpus"] = tokens


def reset_hybrid_bm25_cache() -> None:
    """Clear the cached BM25 index. Call when the dataset reloads."""
    _bm25_cache["bm25"] = None
    _bm25_cache["documents"] = None
    _bm25_cache["tokenized_corpus"] = None


# ─────────────────────────────────────────────────────────────────────
# HybridRetriever
# ─────────────────────────────────────────────────────────────────────

class HybridRetriever(BaseRetriever):
    """Hybrid retriever using Reciprocal Rank Fusion of dense + BM25.

    On each retrieve() call:
      1. Dense path:  asks the vector store for top-K candidates by
                      semantic similarity (FAISS over sentence embeddings).
      2. Sparse path: scores all corpus documents by BM25 over the query.
      3. Fusion:      combines the two ranked lists via Reciprocal Rank
                      Fusion (RRF), producing a single fused ranking.

    The BM25 index is built lazily from the documents the dense store
    holds, on first retrieve() call. This guarantees both retrievers
    see exactly the same corpus — a requirement for fair comparison.
    """

    def __init__(self):
        # Use the same vector store the rest of the platform uses
        self.store = get_vector_store(CONFIG)
        self.store = initialize_store(self.store)

    def _doc_key(self, doc: Any) -> str:
        """Stable identifier for fusion deduplication."""
        if isinstance(doc, dict):
            return doc.get("id") or doc.get("text", "").strip()
        return str(doc).strip()

    def _ensure_bm25(self) -> None:
        """Build the BM25 index from the dense store's corpus, if not yet built."""
        if _bm25_cache["bm25"] is not None:
            return

        # Pull the full corpus from the dense store. Different stores expose
        # this in different ways; we try the documented attributes in order.
        # On FAISSStore, documents are kept in self.documents (list of dicts).
        store_docs = getattr(self.store, "documents", None)
        if store_docs is None:
            # Some stores expose it as ._docs or similar
            store_docs = getattr(self.store, "_documents", None) or getattr(self.store, "_docs", None)

        if not store_docs:
            logger.warning("HybridRetriever: cannot access vector-store corpus for BM25 index.")
            _bm25_cache["bm25"] = None
            _bm25_cache["documents"] = []
            return

        _build_bm25_from_documents(list(store_docs))
        logger.info(f"HybridRetriever: BM25 index built over {len(_bm25_cache['documents'])} documents.")

    def _dense_ranking(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Get the top-k documents from the dense vector store."""
        return self.store.search(query, top_k=k)

    def _sparse_ranking(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Get the top-k documents by BM25 score."""
        if _bm25_cache["bm25"] is None:
            return []

        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        bm25 = _bm25_cache["bm25"]
        documents = _bm25_cache["documents"]
        scores = bm25.get_scores(query_tokens)

        # Rank desc by score, deterministic tiebreak by index
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: (-scores[i], i),
        )[:k]

        return [documents[i] for i in ranked_indices]

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Return the top_k documents fused from dense + BM25 rankings via RRF.

        Args:
            query: User query string.
            top_k: Number of documents to return.

        Returns:
            List of doc dicts of length <= top_k, augmented with a
            "score" field containing the RRF fusion score (descending).
        """
        if top_k < 1:
            raise ValueError(f"top_k must be >= 1, got {top_k}")

        # Build BM25 index on first call (lazy)
        self._ensure_bm25()

        # Get rankings from both retrievers
        dense_docs = self._dense_ranking(query, k=CANDIDATE_POOL_SIZE)
        sparse_docs = self._sparse_ranking(query, k=CANDIDATE_POOL_SIZE)

        # RRF fusion — for each unique document, sum 1/(k + rank) across rankings
        rrf_scores: Dict[str, float] = {}
        doc_lookup: Dict[str, Dict[str, Any]] = {}

        for rank, doc in enumerate(dense_docs, start=1):
            key = self._doc_key(doc)
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_lookup.setdefault(key, doc)

        for rank, doc in enumerate(sparse_docs, start=1):
            key = self._doc_key(doc)
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            doc_lookup.setdefault(key, doc)

        if not rrf_scores:
            return []

        # Sort by fused score descending; deterministic tiebreak by doc key
        ranked_keys = sorted(
            rrf_scores.keys(),
            key=lambda k: (-rrf_scores[k], k),
        )[:top_k]

        results = []
        for key in ranked_keys:
            doc = dict(doc_lookup[key])  # shallow copy
            doc["score"] = float(rrf_scores[key])
            results.append(doc)

        return results
