"""
SimpleReranker — lexical overlap baseline.

Reranks candidate documents by exact-word overlap between the query and
each document. This is intentionally simple — it is positioned in the
ablation as a *lexical baseline* against which the cross-encoder
reranker's neural improvements are measured.

Why a lexical baseline matters:
  - In a 3×3 ablation matrix (3 retrievers × 3 rerankers), one reranker
    cell must be "no rerank" (NoOpReranker) and one cell must be a
    cheap, parameter-free reranker. SimpleReranker fills the latter
    role: it requires no neural model and runs in microseconds.
  - This makes the comparison interpretable: the gap between SimpleReranker
    and CrossEncoderReranker quantifies the *value of neural reranking*
    on top of any base retriever.
  - On well-aligned query/document vocabulary, lexical overlap is a
    reasonable signal. On paraphrase-heavy benchmarks (like FiQA, where
    queries and relevant documents often use different vocabulary), the
    lexical reranker is expected to help less or even hurt — that gap
    is itself a finding for Chapter 6.

Scoring formula:
  score(query, doc) = |query_tokens ∩ doc_tokens|

This is exact-token-set intersection size. It rewards documents that
contain many distinct query terms, regardless of their position or
frequency. Bug-fixed from the prior implementation, which:
  (1) used substring matching on a flattened string (could match
      "the" inside "weather"),
  (2) did not preserve top_k input length,
  (3) had no deterministic tiebreaking.
"""

from __future__ import annotations

import re
from typing import List, Any

from app.rag.base_reranker import BaseReranker


# Standard short English stoplist. Kept short so the score remains
# meaningful — overlong stoplists collapse to noise on short queries.
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "do", "does",
    "for", "from", "has", "have", "how", "in", "is", "it", "of",
    "on", "or", "that", "the", "to", "was", "were", "what", "when",
    "where", "which", "who", "why", "will", "with",
})

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens of length >= 2, with stopword removal."""
    if not text:
        return []
    return [tok.lower() for tok in _TOKEN_RE.findall(text)
            if len(tok) >= 2 and tok.lower() not in STOPWORDS]


class SimpleReranker(BaseReranker):
    """Reranks documents by query-token overlap (lexical baseline).

    Conforms to BaseReranker — only `rerank(query, documents)` is required.

    Scoring:
        score = |set(query_tokens) ∩ set(doc_tokens)|

    Properties:
      - Deterministic: identical inputs → identical outputs
      - O(|query| + |doc|) per document — microsecond latency
      - Returns the same number of documents it received (no truncation)
      - Preserves all original document fields; adds "rerank_score"
      - Stable tiebreaks by original index
    """

    def _doc_text(self, doc: Any) -> str:
        if isinstance(doc, dict):
            return doc.get("text", "")
        return str(doc)

    def rerank(self, query: str, documents: List[Any]) -> List[Any]:
        """Rerank documents by query-token overlap.

        Args:
            query: Query string.
            documents: List of document dicts (or strings).

        Returns:
            New list of documents in descending order of overlap score.
            Length equals input length (no truncation).
            If documents is empty, returns [].
            If query is empty, returns input order unchanged.
        """
        if not documents:
            return []
        if not query or not query.strip():
            # Degenerate case: no query → no reranking signal; preserve order
            return list(documents)

        query_tokens = set(_tokenize(query))
        if not query_tokens:
            return list(documents)

        scored = []
        for i, doc in enumerate(documents):
            doc_tokens = set(_tokenize(self._doc_text(doc)))
            score = len(query_tokens & doc_tokens)

            if isinstance(doc, dict):
                new_doc = dict(doc)  # shallow copy
                if "score" in new_doc and "retrieval_score" not in new_doc:
                    new_doc["retrieval_score"] = new_doc["score"]
                new_doc["rerank_score"] = float(score)
                new_doc["score"] = float(score)  # canonical score = rerank score
            else:
                new_doc = {"text": str(doc), "rerank_score": float(score), "score": float(score)}

            scored.append((score, i, new_doc))

        # Sort by score descending, deterministic tiebreak by original index
        scored.sort(key=lambda x: (-x[0], x[1]))

        return [doc for _, _, doc in scored]
