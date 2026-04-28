"""
Prepare a FiQA evaluation subset with distractor documents.

Constructs an evaluation corpus by combining:
  1. All documents marked relevant to the selected query subset (via qrels)
  2. N randomly sampled distractor documents from the remaining corpus

This addresses a methodological concern with the prior preparation script,
which included only relevant documents. Without distractors, every corpus
document was relevant to *some* query, inflating retrieval recall and
making ablation differences artificially small.

Defensible defaults (stated in Chapter 6 methodology):
  - NUM_QUERIES = 50           : balances statistical power vs. compute time
  - NUM_DISTRACTORS = 500      : ~5x ratio of distractors to relevant docs
  - SEED = 42                  : reproducibility
  - First N queries with qrels : eliminates query-selection bias
  - Distractors sampled with   : reproducible across runs
    pandas random_state=SEED

The 5x ratio is moderate by BEIR convention (corpus-to-query ratios there
typically span 100x-10000x). It produces a corpus difficult enough to
discriminate retriever quality while remaining tractable for nine ablation
runs within the time budget.

For the smoke test, set SMOKE_TEST=1 in the environment to use only 10
queries with 100 distractors. This validates the full pipeline end-to-end
in minutes before committing to the full evaluation pass.

Usage:
    # Full evaluation prep:
    python evaluation/scripts/prepare_fiqa_subset.py

    # Smoke test:
    SMOKE_TEST=1 python evaluation/scripts/prepare_fiqa_subset.py
"""

import os
import json
import pandas as pd
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

OUTPUT_DIR = "evaluation/data/fiqa_subset"
DOCS_DIR = "data/documents"

SMOKE_TEST = os.getenv("SMOKE_TEST", "").strip() in ("1", "true", "yes")

if SMOKE_TEST:
    NUM_QUERIES = 10
    NUM_DISTRACTORS = 100
    print("⚡ SMOKE TEST MODE: 10 queries, 100 distractors")
else:
    NUM_QUERIES = 100
    NUM_DISTRACTORS = 2000
    print("📊 FULL EVALUATION MODE: 100 queries, 2000 distractors")

SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────
# Load FiQA from BEIR
# ─────────────────────────────────────────────────────────────────────

print("Loading FiQA corpus...")
corpus = load_dataset("BeIR/fiqa", "corpus")["corpus"]

print("Loading FiQA queries...")
queries = load_dataset("BeIR/fiqa", "queries")["queries"]

print("Loading FiQA qrels...")
qrels = load_dataset("BeIR/fiqa-qrels")["test"]

corpus_df = pd.DataFrame(corpus)
queries_df = pd.DataFrame(queries)
qrels_df = pd.DataFrame(qrels)

# Normalize ID columns to strings
corpus_df["_id"] = corpus_df["_id"].astype(str)
queries_df["_id"] = queries_df["_id"].astype(str)
qrels_df["query-id"] = qrels_df["query-id"].astype(str)
qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

print(f"Loaded: {len(corpus_df)} corpus docs, {len(queries_df)} queries, {len(qrels_df)} qrels")


# ─────────────────────────────────────────────────────────────────────
# Select queries with full qrels coverage
# ─────────────────────────────────────────────────────────────────────

# We pick the first NUM_QUERIES query IDs that appear in qrels. This is
# deterministic (pandas drop_duplicates preserves insertion order) and
# avoids the bias of picking a random sample of queries — the first N
# represent the head of the qrels file as published.

selected_query_ids = qrels_df["query-id"].drop_duplicates().head(NUM_QUERIES).tolist()
selected_qrels = qrels_df[qrels_df["query-id"].isin(selected_query_ids)]

# Documents that are relevant to AT LEAST ONE selected query
relevant_doc_ids = selected_qrels["corpus-id"].drop_duplicates().tolist()
relevant_docs = corpus_df[corpus_df["_id"].isin(relevant_doc_ids)]

print(f"Selected: {len(selected_query_ids)} queries with {len(relevant_doc_ids)} relevant docs")


# ─────────────────────────────────────────────────────────────────────
# Sample distractor documents
# ─────────────────────────────────────────────────────────────────────

# Distractors are FiQA corpus documents that are NOT relevant to any
# selected query. They are sampled deterministically with random_state=SEED
# to support reproducibility.

available_distractors = corpus_df[~corpus_df["_id"].isin(relevant_doc_ids)]

# Cap NUM_DISTRACTORS to what's available (defensive — FiQA has 57K, so
# this should never be a real constraint, but useful for smoke tests).
n_to_sample = min(NUM_DISTRACTORS, len(available_distractors))

distractor_docs = available_distractors.sample(n=n_to_sample, random_state=SEED)

print(f"Sampled {len(distractor_docs)} distractor docs (seed={SEED})")


# ─────────────────────────────────────────────────────────────────────
# Combine relevant + distractors → final corpus
# ─────────────────────────────────────────────────────────────────────

selected_docs = pd.concat([relevant_docs, distractor_docs], ignore_index=True)

print(f"\n📦 Final evaluation corpus: {len(selected_docs)} documents")
print(f"   = {len(relevant_docs)} relevant + {len(distractor_docs)} distractors")


# ─────────────────────────────────────────────────────────────────────
# Select queries
# ─────────────────────────────────────────────────────────────────────

selected_queries = queries_df[queries_df["_id"].isin(selected_query_ids)]
print(f"📝 Final query set: {len(selected_queries)} queries")


# ─────────────────────────────────────────────────────────────────────
# Save evaluation files (queries.csv, qrels.csv, documents.csv)
# ─────────────────────────────────────────────────────────────────────

selected_queries.rename(columns={"_id": "query_id", "text": "query"})[
    ["query_id", "query"]
].to_csv(os.path.join(OUTPUT_DIR, "queries.csv"), index=False)

selected_qrels.rename(
    columns={
        "query-id": "query_id",
        "corpus-id": "doc_id",
        "score": "relevance",
    }
)[["query_id", "doc_id", "relevance"]].to_csv(
    os.path.join(OUTPUT_DIR, "qrels.csv"),
    index=False,
)

selected_docs.rename(columns={"_id": "doc_id"})[
    ["doc_id", "title", "text"]
].to_csv(os.path.join(OUTPUT_DIR, "documents.csv"), index=False)


# ─────────────────────────────────────────────────────────────────────
# Write documents to data/documents/ for the platform's ingestion pipeline
# ─────────────────────────────────────────────────────────────────────

# Clear previous documents folder so old runs don't contaminate the corpus
print(f"\nClearing {DOCS_DIR}/ ...")
for filename in os.listdir(DOCS_DIR):
    path = os.path.join(DOCS_DIR, filename)
    if os.path.isfile(path):
        os.remove(path)

# Write each document as fiqa_<doc_id>.txt (filename pattern matches
# the prior convention so run_retrieval_eval.py's normalize_doc_id
# regex still works without modification).
print(f"Writing {len(selected_docs)} documents to {DOCS_DIR}/ ...")
for _, row in selected_docs.iterrows():
    doc_id = str(row["_id"])
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""

    content = f"Document ID: {doc_id}\nTitle: {title}\n\n{text}"

    with open(os.path.join(DOCS_DIR, f"fiqa_{doc_id}.txt"), "w", encoding="utf-8") as f:
        f.write(content)


# ─────────────────────────────────────────────────────────────────────
# Save reproducibility metadata
# ─────────────────────────────────────────────────────────────────────

metadata = {
    "smoke_test": SMOKE_TEST,
    "num_queries": NUM_QUERIES,
    "num_relevant_docs": len(relevant_docs),
    "num_distractors": len(distractor_docs),
    "num_total_docs": len(selected_docs),
    "seed": SEED,
    "fiqa_full_corpus_size": len(corpus_df),
    "selected_query_ids_first_5": selected_query_ids[:5],
}
with open(os.path.join(OUTPUT_DIR, "preparation_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)


print(f"\n✅ Done.")
print(f"   Evaluation files: {OUTPUT_DIR}/")
print(f"   Document corpus:  {DOCS_DIR}/ ({len(selected_docs)} files)")
print(f"   Metadata:         {OUTPUT_DIR}/preparation_metadata.json")
print(f"\nNext: run `python evaluation/scripts/run_retrieval_eval.py`")
