import math
import os
import re
import time
from collections import defaultdict

import pandas as pd
import requests


RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8002").rstrip("/")

DATA_DIR = "evaluation/data/fiqa_subset"
RESULTS_DIR = "evaluation/results"

QUERIES_PATH = os.path.join(DATA_DIR, "queries.csv")
QRELS_PATH = os.path.join(DATA_DIR, "qrels.csv")

DETAILED_RESULTS_PATH = os.path.join(RESULTS_DIR, "retrieval_detailed_results.csv")
SUMMARY_RESULTS_PATH = os.path.join(RESULTS_DIR, "retrieval_summary_results.csv")

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


def normalize_doc_id(raw_id: str) -> str:
    """
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


def reload_dataset(retriever: str, reranker: str, top_k: int):
    payload = {
        "document_source": "local",
        "vector_store": "faiss",
        "retriever": retriever,
        "reranker": reranker,
        "generator": "mock",
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
    response = requests.post(
        f"{RAG_API_URL}/search",
        json={"query": query, "top_k": top_k},
        timeout=300,
    )
    response.raise_for_status()
    return response.json()


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
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def dcg_at_k(retrieved, relevant_scores, k):
    score = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        rel = relevant_scores.get(doc_id, 0)
        if rel > 0:
            score += (2**rel - 1) / math.log2(rank + 1)
    return score


def ndcg_at_k(retrieved, relevant_scores, k):
    actual_dcg = dcg_at_k(retrieved, relevant_scores, k)

    ideal_relevances = sorted(relevant_scores.values(), reverse=True)[:k]
    ideal_dcg = 0.0
    for rank, rel in enumerate(ideal_relevances, start=1):
        ideal_dcg += (2**rel - 1) / math.log2(rank + 1)

    if ideal_dcg == 0:
        return 0.0

    return actual_dcg / ideal_dcg


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    queries_df = pd.read_csv(QUERIES_PATH)
    qrels_df = pd.read_csv(QRELS_PATH)

    queries_df["query_id"] = queries_df["query_id"].astype(str)
    qrels_df["query_id"] = qrels_df["query_id"].astype(str)
    qrels_df["doc_id"] = qrels_df["doc_id"].astype(str)

    qrels_by_query = defaultdict(dict)
    for _, row in qrels_df.iterrows():
        qrels_by_query[row["query_id"]][row["doc_id"]] = int(row.get("relevance", 1))

    detailed_rows = []

    for retriever, reranker in CONFIGS:
        max_k = max(TOP_K_VALUES)

        print(f"\n=== Config: retriever={retriever}, reranker={reranker} ===")
        reload_result = reload_dataset(retriever, reranker, max_k)
        print(f"Reloaded dataset: {reload_result}")

        for _, row in queries_df.iterrows():
            query_id = str(row["query_id"])
            query = str(row["query"])

            relevant_scores = qrels_by_query.get(query_id, {})
            relevant_doc_ids = set(relevant_scores.keys())

            start = time.perf_counter()
            result = search(query, max_k)
            elapsed_ms = round((time.perf_counter() - start) * 1000, 2)

            retrieved_docs = result.get("documents", [])
            retrieved_ids = [
                normalize_doc_id(doc.get("id", ""))
                for doc in retrieved_docs
                if isinstance(doc, dict)
            ]

            latency = result.get("latency", {})

            for k in TOP_K_VALUES:
                detailed_rows.append(
                    {
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
                        "client_elapsed_ms": elapsed_ms,
                    }
                )

            print(f"Query {query_id}: retrieved {retrieved_ids[:max_k]}")

    detailed_df = pd.DataFrame(detailed_rows)
    detailed_df.to_csv(DETAILED_RESULTS_PATH, index=False)

    summary_df = (
        detailed_df.groupby(["retriever", "reranker", "k"], as_index=False)
        .agg(
            precision_at_k=("precision_at_k", "mean"),
            recall_at_k=("recall_at_k", "mean"),
            mrr_at_k=("mrr_at_k", "mean"),
            ndcg_at_k=("ndcg_at_k", "mean"),
            retrieval_ms=("retrieval_ms", "mean"),
            reranking_ms=("reranking_ms", "mean"),
            backend_total_ms=("backend_total_ms", "mean"),
            client_elapsed_ms=("client_elapsed_ms", "mean"),
        )
    )

    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    print("\nSaved:")
    print(DETAILED_RESULTS_PATH)
    print(SUMMARY_RESULTS_PATH)

    print("\nSummary:")
    print(summary_df.sort_values(["k", "ndcg_at_k", "recall_at_k"], ascending=[True, False, False]))


if __name__ == "__main__":
    main()