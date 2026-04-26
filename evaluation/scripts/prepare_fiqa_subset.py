import os
import json
import pandas as pd
from datasets import load_dataset

OUTPUT_DIR = "evaluation/data/fiqa_subset"
DOCS_DIR = "data/documents"

NUM_QUERIES = 10

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)

print("Loading FiQA corpus...")
corpus = load_dataset("BeIR/fiqa", "corpus")["corpus"]

print("Loading FiQA queries...")
queries = load_dataset("BeIR/fiqa", "queries")["queries"]

print("Loading FiQA qrels...")
qrels = load_dataset("BeIR/fiqa-qrels")["test"]

corpus_df = pd.DataFrame(corpus)
queries_df = pd.DataFrame(queries)
qrels_df = pd.DataFrame(qrels)

# Normalize column names
corpus_df["_id"] = corpus_df["_id"].astype(str)
queries_df["_id"] = queries_df["_id"].astype(str)
qrels_df["query-id"] = qrels_df["query-id"].astype(str)
qrels_df["corpus-id"] = qrels_df["corpus-id"].astype(str)

# Pick first NUM_QUERIES with relevance judgments
selected_query_ids = qrels_df["query-id"].drop_duplicates().head(NUM_QUERIES).tolist()
selected_qrels = qrels_df[qrels_df["query-id"].isin(selected_query_ids)]

selected_doc_ids = selected_qrels["corpus-id"].drop_duplicates().tolist()

selected_queries = queries_df[queries_df["_id"].isin(selected_query_ids)]
selected_docs = corpus_df[corpus_df["_id"].isin(selected_doc_ids)]

print(f"Selected queries: {len(selected_queries)}")
print(f"Selected relevant documents: {len(selected_docs)}")

# Save query/qrels files for evaluation
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

# Clear current docs folder before writing FiQA subset
for filename in os.listdir(DOCS_DIR):
    path = os.path.join(DOCS_DIR, filename)
    if os.path.isfile(path):
        os.remove(path)

# Write each FiQA document as a .txt file for your existing ingestion pipeline
for _, row in selected_docs.iterrows():
    doc_id = str(row["_id"])
    title = row.get("title", "") or ""
    text = row.get("text", "") or ""

    content = f"Document ID: {doc_id}\nTitle: {title}\n\n{text}"

    with open(os.path.join(DOCS_DIR, f"fiqa_{doc_id}.txt"), "w", encoding="utf-8") as f:
        f.write(content)

print(f"Wrote FiQA documents to: {DOCS_DIR}")
print(f"Saved evaluation files to: {OUTPUT_DIR}")