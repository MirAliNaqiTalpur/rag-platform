"""
Stratify the FiQA evaluation queries for LLM-as-a-Judge sampling.

Selects queries from the retrieval evaluation set, stratified by retrieval
difficulty. Stratification is based on per-query nDCG@10 from the simple+none
configuration. The default full sample is:

    Easy   (nDCG@10 >= 0.80): 20 queries
    Medium (0.40 < nDCG < 0.80): 20 queries
    Hard   (nDCG@10 <= 0.40): 10 queries

Why stratification rather than plain random sampling:
  - Random sampling can overrepresent whichever stratum is most populated.
  - Stratified sampling gives a clearer view of performance across easy,
    medium, and hard cases.
  - The stratum boundaries are taken from simple+none to define difficulty
    consistently before comparing reranking configurations.

Sampling within each stratum uses a fixed random seed for reproducibility.

Usage:
    python evaluation/scripts/stratify_queries.py

Smoke-test usage:
    JUDGE_N_EASY=1 JUDGE_N_MEDIUM=1 JUDGE_N_HARD=1 python evaluation/scripts/stratify_queries.py

Output:
    evaluation/data/fiqa_subset/judge_query_subset.csv
        columns: query_id, query, stratum, ndcg_at_10_simple_none
"""

import os

import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

DETAILED_PATH = os.getenv(
    "RETRIEVAL_DETAILED_PATH",
    "evaluation/results/retrieval_detailed_results.csv",
)
OUTPUT_PATH = os.getenv(
    "JUDGE_QUERY_SUBSET_PATH",
    "evaluation/data/fiqa_subset/judge_query_subset.csv",
)

EASY_THRESHOLD = float(os.getenv("JUDGE_EASY_THRESHOLD", "0.80"))
HARD_THRESHOLD = float(os.getenv("JUDGE_HARD_THRESHOLD", "0.40"))

N_EASY = int(os.getenv("JUDGE_N_EASY", "20"))
N_MEDIUM = int(os.getenv("JUDGE_N_MEDIUM", "20"))
N_HARD = int(os.getenv("JUDGE_N_HARD", "10"))
RANDOM_SEED = int(os.getenv("JUDGE_SAMPLE_SEED", "42"))


def main():
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

    if not os.path.exists(DETAILED_PATH):
        raise SystemExit(
            f"Missing {DETAILED_PATH}. Run retrieval evaluation first, or set "
            "RETRIEVAL_DETAILED_PATH to the correct file."
        )

    df = pd.read_csv(DETAILED_PATH)
    required_cols = {"retriever", "reranker", "k", "query_id", "query", "ndcg_at_k"}
    missing = required_cols - set(df.columns)
    if missing:
        raise SystemExit(f"{DETAILED_PATH} is missing columns: {sorted(missing)}")

    baseline = df[
        (df["retriever"] == "simple")
        & (df["reranker"] == "none")
        & (df["k"] == 10)
    ][["query_id", "query", "ndcg_at_k"]].rename(
        columns={"ndcg_at_k": "ndcg_at_10_simple_none"}
    )

    if baseline.empty:
        raise SystemExit(
            "No simple+none rows at k=10 found in retrieval_detailed_results.csv. "
            "Check retriever/reranker/k labels."
        )

    baseline["query_id"] = baseline["query_id"].astype(str)
    baseline["query_id_sort"] = pd.to_numeric(baseline["query_id"], errors="coerce")
    baseline = baseline.sort_values(["query_id_sort", "query_id"]).drop(
        columns=["query_id_sort"]
    ).reset_index(drop=True)

    def label(ndcg):
        if ndcg >= EASY_THRESHOLD:
            return "easy"
        if ndcg <= HARD_THRESHOLD:
            return "hard"
        return "medium"

    baseline["stratum"] = baseline["ndcg_at_10_simple_none"].apply(label)

    print(f"Stratum populations from {len(baseline)} queries:")
    print(baseline["stratum"].value_counts().to_string())
    print()

    samples = []
    for stratum, n in [("easy", N_EASY), ("medium", N_MEDIUM), ("hard", N_HARD)]:
        pool = baseline[baseline["stratum"] == stratum]
        if len(pool) < n:
            print(
                f"WARNING: requested {n} {stratum} queries but only "
                f"{len(pool)} available; using all."
            )
            n = len(pool)
        if n > 0:
            samples.append(pool.sample(n=n, random_state=RANDOM_SEED))

    if not samples:
        raise SystemExit("No queries selected. Check sample-size environment variables.")

    selected = pd.concat(samples, ignore_index=True)
    stratum_order = {"easy": 0, "medium": 1, "hard": 2}
    selected["stratum_order"] = selected["stratum"].map(stratum_order)
    selected["query_id_sort"] = pd.to_numeric(selected["query_id"], errors="coerce")
    selected = selected.sort_values(["stratum_order", "query_id_sort", "query_id"]).drop(
        columns=["stratum_order", "query_id_sort"]
    ).reset_index(drop=True)

    selected = selected[["query_id", "query", "stratum", "ndcg_at_10_simple_none"]]
    selected.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved {len(selected)} queries to {OUTPUT_PATH}")
    print(f"  Easy:   {(selected['stratum'] == 'easy').sum()}")
    print(f"  Medium: {(selected['stratum'] == 'medium').sum()}")
    print(f"  Hard:   {(selected['stratum'] == 'hard').sum()}")
    print(f"  Seed:   {RANDOM_SEED}")


if __name__ == "__main__":
    main()
