"""
Retrieval evaluation harness.

Runs all 9 retriever × reranker configurations against the platform's
/search endpoint and computes standard IR metrics (Recall@K, Precision@K,
MRR@K, nDCG@K) per query.

Methodological notes:
  - WARMUP: each configuration runs one warmup query before the timed
    loop. This excludes cold-start index-build cost (lazy initialization
    of BM25 corpus, dense embedding model load) from latency measurements.
    Without warmup, the first query's index-build time is ~50× the
    steady-state per-query latency and would dominate the mean,
    producing misleading Chapter 6 results.
  - LATENCY STATS: mean, median, P95, and max are all reported. Median
    is the recommended summary because cold-start outliers and external
    noise (GC pauses, etc.) skew the mean. P95 captures tail behavior.
  - PAIRED COMPARISON: same query set across all configs → enables paired
    statistical tests in downstream analysis.
  - FIXED ORDER: configs are run in deterministic order; queries within
    each config are run in deterministic order (CSV row order).

Outputs:
  - retrieval_detailed_results.csv : per-config, per-query, per-K row
                                     with all metrics + latencies
  - retrieval_summary_results.csv  : per-config, per-K aggregates
                                     (mean + median + p95 + max for latency,
                                      mean for retrieval metrics)
  - retrieval_run_metadata.json    : run config + timestamps for reproducibility
"""

import json
import math
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone

import pandas as pd
import requests


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8002").rstrip("/")

DATA_DIR = "evaluation/data/fiqa_subset"
RESULTS_DIR = "evaluation/results"

QUERIES_PATH = os.path.join(DATA_DIR, "queries.csv")
QRELS_PATH = os.path.join(DATA_DIR, "qrels.csv")

DETAILED_RESULTS_PATH = os.path.join(RESULTS_DIR, "retrieval_detailed_results.csv")
SUMMARY_RESULTS_PATH = os.path.join(RESULTS_DIR, "retrieval_summary_results.csv")
METADATA_PATH = os.path.join(RESULTS_DIR, "retrieval_run_metadata.json")

TOP_K_VALUES = [3, 5, 10]

CONFIGS = [
    ("bm25", "none"),
    ("bm25", "simple"),
    ("bm25", "cross_encoder"),
    ("simple", "none"),
    ("simple", "simple"),
    ("simple", "cross_encoder"),
    ("hybrid", "none"),
    ("hybrid", "simple"),
    ("hybrid", "cross_encoder"),
]

# Warmup query — generic financial question used to trigger index builds
# and model loads before timing starts. Not part of the evaluation set
# (so it doesn't bias measurements). Same query across all configs for
# consistency.
WARMUP_QUERY = "What are the tax implications of investments?"


# ─────────────────────────────────────────────────────────────────────
# Doc-ID normalization
# ─────────────────────────────────────────────────────────────────────

def normalize_doc_id(raw_id: str) -> str:
    """Strip platform's filename wrapper to recover the original FiQA doc ID.

    Converts:
        data/documents/fiqa_31793.txt -> 31793
        fiqa_31793.txt -> 31793
        31793 -> 31793
    """
    if raw_id is None:
        return ""

    raw_id = str(raw_id)
    match = re.search(r"fiqa_(\d+)\.txt", raw_id)
    if match:
        return match.group(1)

    match = re.search(r"(\d+)", raw_id)
    if match:
        return match.group(1)

    return raw_id.strip()


# ─────────────────────────────────────────────────────────────────────
# Platform API calls
# ─────────────────────────────────────────────────────────────────────

def reload_dataset(retriever: str, reranker: str, top_k: int):
    """Switch the platform to a new (retriever, reranker) configuration."""
    payload = {
        "document_source": "local",
        "vector_store": "faiss",
        "retriever": retriever,
        "reranker": reranker,
        "generator": "mock",  # generation not needed for retrieval eval
        "top_k": top_k,
    }

    response = requests.post(
        f"{RAG_API_URL}/reload-dataset",
        json=payload,
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


def search(query: str, top_k: int):
    """Run a search query against /search endpoint."""
    response = requests.post(
        f"{RAG_API_URL}/search",
        json={"query": query, "top_k": top_k},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


# ─────────────────────────────────────────────────────────────────────
# IR metrics (pure-Python, matches pytrec_eval semantics)
# ─────────────────────────────────────────────────────────────────────

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    if not retrieved_k:
        return 0.0
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / k


def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    retrieved_k = retrieved[:k]
    hits = sum(1 for doc_id in retrieved_k if doc_id in relevant)
    return hits / len(relevant)


def mrr_at_k(retrieved, relevant, k):
    """Mean Reciprocal Rank at K — 1/rank of first relevant doc, else 0."""
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved, relevant_scores, k):
    """Discounted Cumulative Gain at K, using gain = 2^rel - 1."""
    score = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevant_scores.get(doc_id, 0)
        if rel > 0:
            score += (2**rel - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved, relevant_scores, k):
    """Normalized DCG at K = DCG@K / ideal DCG@K."""
    actual_dcg = dcg_at_k(retrieved, relevant_scores, k)

    ideal_relevances = sorted(relevant_scores.values(), reverse=True)[:k]
    ideal_dcg = 0.0
    for rank, rel in enumerate(ideal_relevances, start=1):
        ideal_dcg += (2**rel - 1) / math.log2(rank + 1)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


# ─────────────────────────────────────────────────────────────────────
# Latency aggregation helpers
# ─────────────────────────────────────────────────────────────────────

def percentile(values, p):
    """Compute the p-th percentile (0 < p < 100) using linear interpolation."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    k = (len(sorted_vals) - 1) * (p / 100.0)
    lo = int(math.floor(k))
    hi = int(math.ceil(k))
    if lo == hi:
        return float(sorted_vals[lo])
    return float(sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (k - lo))


# ─────────────────────────────────────────────────────────────────────
# Main evaluation loop
# ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Load eval data
    queries_df = pd.read_csv(QUERIES_PATH)
    qrels_df = pd.read_csv(QRELS_PATH)

    queries_df["query_id"] = queries_df["query_id"].astype(str)
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    qrels_by_query = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels_by_query[row["query_id"]][row["doc_id"]] = int(row.get("relevance", 1))

    print(f"Loaded {len(queries_df)} queries with qrels for "
          f"{len(qrels_by_query)} queries")

    # Run metadata
    run_start_iso = datetime.now(timezone.utc).isoformat()

    detailed_rows = []
    cold_start_records = []  # tracks first-query latency per config
    config_durations = {}  # wall-clock per config

    for retriever, reranker in CONFIGS:
        max_k = max(TOP_K_VALUES)
        config_label = f"{retriever}+{reranker}"

        print(f"\n{'='*70}")
        print(f"Config: retriever={retriever}, reranker={reranker}")
        print(f"{'='*70}")

        config_start = time.perf_counter()

        # Switch the platform's configuration
        reload_result = reload_dataset(retriever, reranker, max_k)
        print(f"  Reload-dataset OK: {reload_result.get('status', reload_result)}")

        # ─────────────────────────────────────────────────────────
        # WARMUP — exclude cold-start cost from timed measurements
        # ─────────────────────────────────────────────────────────
        # Lazy index builds (BM25 corpus tokenization, embedding model
        # load) happen on the first /search call. We run one untimed
        # warmup query so the timed loop measures steady-state behavior.
        warmup_start = time.perf_counter()
        try:
            warmup_result = search(WARMUP_QUERY, max_k)
            warmup_ms = (time.perf_counter() - warmup_start) * 1000
            warmup_total_ms = warmup_result.get("latency", {}).get("total_ms", warmup_ms)
            print(f"  Warmup query: {warmup_total_ms:.1f} ms (excluded from results)")
            cold_start_records.append({
                "retriever": retriever,
                "reranker": reranker,
                "warmup_total_ms": float(warmup_total_ms),
            })
        except Exception as e:
            print(f"  ⚠ Warmup failed: {e}")
            cold_start_records.append({
                "retriever": retriever,
                "reranker": reranker,
                "warmup_total_ms": None,
            })

        # ─────────────────────────────────────────────────────────
        # Timed evaluation loop
        # ─────────────────────────────────────────────────────────
        for _, row in queries_df.iterrows():
            query_id = str(row["query_id"])
            query = str(row["query"])

            relevant_scores = qrels_by_query.get(query_id, {})
            relevant_doc_ids = set(relevant_scores.keys())

            client_start = time.perf_counter()
            result = search(query, max_k)
            client_elapsed_ms = round((time.perf_counter() - client_start) * 1000, 2)

            retrieved_docs = result.get("documents", [])
            retrieved_ids = [
                normalize_doc_id(doc.get("id", ""))
                for doc in retrieved_docs
                if isinstance(doc, dict)
            ]

            latency = result.get("latency", {})

            for k in TOP_K_VALUES:
                detailed_rows.append({
                    "retriever": retriever,
                    "reranker": reranker,
                    "query_id": query_id,
                    "query": query,
                    "k": k,
                    "relevant_doc_ids": ";".join(sorted(relevant_doc_ids)),
                    "retrieved_doc_ids": ";".join(retrieved_ids[:k]),
                    "precision_at_k": precision_at_k(retrieved_ids, relevant_doc_ids, k),
                    "recall_at_k": recall_at_k(retrieved_ids, relevant_doc_ids, k),
                    "mrr_at_k": mrr_at_k(retrieved_ids, relevant_doc_ids, k),
                    "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant_scores, k),
                    "retrieval_ms": latency.get("retrieval_ms"),
                    "reranking_ms": latency.get("reranking_ms"),
                    "backend_total_ms": latency.get("total_ms"),
                    "client_elapsed_ms": client_elapsed_ms,
                })

        config_durations[config_label] = round(time.perf_counter() - config_start, 2)
        print(f"  Config wall-clock: {config_durations[config_label]} sec")

    # ─────────────────────────────────────────────────────────
    # Aggregate to summary
    # ─────────────────────────────────────────────────────────
    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(DETAILED_RESULTS_PATH, index=False)

    # For metrics: take mean across queries.
    # For latencies: report mean, median, P95, max separately.
    summary_rows = []
    for (retriever, reranker, k), group in detailed_df.groupby(
        ["retriever", "reranker", "k"]
    ):
        summary_rows.append({
            "retriever": retriever,
            "reranker": reranker,
            "k": k,
            # IR metrics — mean across queries
            "precision_at_k": group["precision_at_k"].mean(),
            "recall_at_k": group["recall_at_k"].mean(),
            "mrr_at_k": group["mrr_at_k"].mean(),
            "ndcg_at_k": group["ndcg_at_k"].mean(),
            # Latency — mean for back-compat, median preferred for steady-state,
            # p95 for tail, max for worst case
            "retrieval_ms_mean": group["retrieval_ms"].mean(),
            "retrieval_ms_median": group["retrieval_ms"].median(),
            "retrieval_ms_p95": percentile(group["retrieval_ms"].dropna().tolist(), 95),
            "retrieval_ms_max": group["retrieval_ms"].max(),
            "reranking_ms_mean": group["reranking_ms"].mean(),
            "reranking_ms_median": group["reranking_ms"].median(),
            "reranking_ms_p95": percentile(group["reranking_ms"].dropna().tolist(), 95),
            "reranking_ms_max": group["reranking_ms"].max(),
            "backend_total_ms_mean": group["backend_total_ms"].mean(),
            "backend_total_ms_median": group["backend_total_ms"].median(),
            "backend_total_ms_p95": percentile(group["backend_total_ms"].dropna().tolist(), 95),
            "backend_total_ms_max": group["backend_total_ms"].max(),
            "client_elapsed_ms_mean": group["client_elapsed_ms"].mean(),
            "client_elapsed_ms_median": group["client_elapsed_ms"].median(),
            "n_queries": int(len(group)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values(
        ["k", "ndcg_at_k", "recall_at_k"],
        ascending=[True, False, False],
    ).reset_index(drop=True)
    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    # ─────────────────────────────────────────────────────────
    # Run metadata for reproducibility
    # ─────────────────────────────────────────────────────────
    metadata = {
        "run_start_utc": run_start_iso,
        "run_end_utc": datetime.now(timezone.utc).isoformat(),
        "rag_api_url": RAG_API_URL,
        "n_queries": int(len(queries_df)),
        "configs_evaluated": [list(c) for c in CONFIGS],
        "top_k_values": TOP_K_VALUES,
        "warmup_query": WARMUP_QUERY,
        "warmup_excluded": True,
        "cold_start_records": cold_start_records,
        "config_durations_sec": config_durations,
        "metrics": ["precision_at_k", "recall_at_k", "mrr_at_k", "ndcg_at_k"],
        "latency_aggregations": ["mean", "median", "p95", "max"],
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*70}")
    print("Saved:")
    print(f"  {DETAILED_RESULTS_PATH}")
    print(f"  {SUMMARY_RESULTS_PATH}")
    print(f"  {METADATA_PATH}")

    # Print compact summary at K=10 for quick eyeball check
    print(f"\n{'='*70}")
    print("HEADLINE — at K=10, sorted by nDCG")
    print(f"{'='*70}")
    print(f"{'Retriever':<10} {'Reranker':<14} {'Recall':>8} {'MRR':>7} {'nDCG':>7} "
          f"{'Median ms':>10}")
    print("-" * 70)
    summary_at_10 = summary_df[summary_df["k"] == 10].sort_values("ndcg_at_k", ascending=False)
    for _, r in summary_at_10.iterrows():
        print(f"{r['retriever']:<10} {r['reranker']:<14} "
              f"{r['recall_at_k']:>8.3f} {r['mrr_at_k']:>7.3f} {r['ndcg_at_k']:>7.3f} "
              f"{r['backend_total_ms_median']:>10.1f}")


if __name__ == "__main__":
    main()
