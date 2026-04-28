"""
LLM-as-a-Judge evaluation harness.

Generates answers from the platform under each evaluated configuration, then
scores those answers using Gemini 2.5 Pro as a judge.

Evaluation design:
  - Generator model: configurable through DEFAULT_MODEL
                     (default: gemini-2.5-flash-lite)
  - Judge model: Gemini 2.5 Pro at temperature 0
  - Configurations evaluated: simple+none, simple+cross_encoder
  - Query subset: stratified FiQA subset created by stratify_queries.py
  - Repetitions: configurable through JUDGE_REPETITIONS
                 (default: 3 judge passes per query/config)
  - Rubric: Groundedness, Relevance, Completeness on 1-5 ordinal scales

Interpretation note:
  This evaluation assesses answer quality conditional on the retrieved context
  returned by each RAG configuration. It does not directly compare generated
  answers against a human-written reference answer.

Why repeated judge passes:
  Reporting repeated judge passes provides an empirical check on judge-score
  stability, including possible non-determinism from the model or API service.

Output files:
  evaluation/results/judge_raw_responses.jsonl
      Per-call records: query_id, config, repetition, judge prompt, raw response,
      parsed scores, latency. One JSON object per line.
  evaluation/results/judge_summary.csv
      Per-(query_id, config) aggregates: mean and stddev of each score
      across repetitions.
  evaluation/results/judge_aggregate.csv
      Per-config summary: mean of per-query mean scores, and stratum-conditional
      means (easy / medium / hard / all).
  evaluation/results/judge_run_metadata.json
      Run timestamps, models used, error counts, and sample metadata.

Usage from the rag-platform repo root:
    python evaluation/scripts/stratify_queries.py
    python evaluation/scripts/llm_judge.py

Useful smoke-test environment variables:
    JUDGE_REPETITIONS=1
    RAG_API_URL=http://localhost:8002
    DEFAULT_MODEL=gemini-2.5-flash-lite

Ensure GEMINI_API_KEY is set in the environment, and the platform is running
locally so /query is reachable.
"""

import json
import math
import os
import re
import time
from datetime import datetime, timezone

import pandas as pd
import requests
from google import genai

try:
    from google.genai import types as genai_types
except Exception:  # pragma: no cover - fallback for older google-genai versions
    genai_types = None


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RAG_API_URL = os.getenv("RAG_API_URL", "http://localhost:8002").rstrip("/")

DATA_DIR = "evaluation/data/fiqa_subset"
RESULTS_DIR = "evaluation/results"
QUERY_SUBSET_PATH = os.path.join(DATA_DIR, "judge_query_subset.csv")

RAW_OUT = os.path.join(RESULTS_DIR, "judge_raw_responses.jsonl")
SUMMARY_OUT = os.path.join(RESULTS_DIR, "judge_summary.csv")
AGGREGATE_OUT = os.path.join(RESULTS_DIR, "judge_aggregate.csv")
METADATA_OUT = os.path.join(RESULTS_DIR, "judge_run_metadata.json")

GENERATOR_MODEL = os.getenv("DEFAULT_MODEL", "gemini-2.5-flash-lite")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemini-2.5-pro")

CONFIGS = [
    ("simple", "none"),
    ("simple", "cross_encoder"),
]

N_REPETITIONS = int(os.getenv("JUDGE_REPETITIONS", "3"))
TOP_K = int(os.getenv("JUDGE_TOP_K", "5"))
JUDGE_MAX_OUTPUT_TOKENS = int(os.getenv("JUDGE_MAX_OUTPUT_TOKENS", "4096"))
JUDGE_MAX_CONTEXT_CHARS_PER_DOC = int(os.getenv("JUDGE_MAX_CONTEXT_CHARS_PER_DOC", "1800"))
JUDGE_RETRY_PLAIN_ON_EMPTY = os.getenv("JUDGE_RETRY_PLAIN_ON_EMPTY", "true").strip().lower() == "true"


# ─────────────────────────────────────────────────────────────────────
# Judge prompt — fixed across all calls for consistency
# ─────────────────────────────────────────────────────────────────────

JUDGE_PROMPT_TEMPLATE = """You are an expert evaluator of question-answering systems. \
Score the answer below on three dimensions, using only the information in the question, \
the retrieved context, and the answer.

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GENERATED ANSWER:
{answer}

Rate the answer on these three rubrics on a 1-5 ordinal scale.

GROUNDEDNESS (1-5): The extent to which every claim in the answer is supported by \
the retrieved context. A score of 5 means every factual claim is directly supported. \
A score of 1 means the answer makes claims that contradict or have no support in the \
context.

RELEVANCE (1-5): The extent to which the answer actually addresses the user's question. \
A score of 5 means the answer directly addresses what was asked. A score of 1 means \
the answer is on a different topic.

COMPLETENESS (1-5): The extent to which the answer covers the key information needed \
to fully answer the question. A score of 5 means the answer is comprehensive given the \
context. A score of 1 means the answer is severely incomplete.

Output your scores as STRICT JSON in exactly this format, with no additional text:

{{
  "groundedness": <integer 1-5>,
  "relevance": <integer 1-5>,
  "completeness": <integer 1-5>,
  "groundedness_reason": "<brief, one-sentence justification>",
  "relevance_reason": "<brief, one-sentence justification>",
  "completeness_reason": "<brief, one-sentence justification>"
}}"""


# ─────────────────────────────────────────────────────────────────────
# Platform calls
# ─────────────────────────────────────────────────────────────────────

def reload_dataset(retriever, reranker, top_k):
    """Switch the platform to a configuration."""
    payload = {
        "document_source": "local",
        "vector_store": "faiss",
        "retriever": retriever,
        "reranker": reranker,
        "generator": "gemini",
        "top_k": top_k,
    }
    r = requests.post(f"{RAG_API_URL}/reload-dataset", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


def query_platform(question, top_k):
    """Run a /query call and get back answer + retrieved docs."""
    payload = {
        "query": question,
        "top_k": top_k,
        "model": GENERATOR_MODEL,
    }
    r = requests.post(f"{RAG_API_URL}/query", json=payload, timeout=300)
    r.raise_for_status()
    return r.json()


# ─────────────────────────────────────────────────────────────────────
# Judge JSON parsing — robust against minor format drift
# ─────────────────────────────────────────────────────────────────────

def parse_judge_response(text):
    """Extract scores from the judge response text.

    Returns a dict with keys: groundedness, relevance, completeness,
    groundedness_reason, relevance_reason, completeness_reason.
    Returns None if the response cannot be parsed.
    """
    if not text:
        return None

    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = text[start:end + 1]

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        candidate2 = re.sub(r",\s*}", "}", candidate)
        candidate2 = re.sub(r",\s*]", "]", candidate2)
        try:
            parsed = json.loads(candidate2)
        except json.JSONDecodeError:
            return None

    out = {}
    for key in ("groundedness", "relevance", "completeness"):
        val = parsed.get(key)
        if isinstance(val, (int, float)):
            out[key] = max(1, min(5, int(val)))
        else:
            return None

    for key in ("groundedness_reason", "relevance_reason", "completeness_reason"):
        out[key] = str(parsed.get(key, ""))[:500]

    return out


# ─────────────────────────────────────────────────────────────────────
# Judge API call
# ─────────────────────────────────────────────────────────────────────

def build_judge_config(json_mode=True):
    """Build Gemini judge config. Keep max_output_tokens high for Gemini 2.5 models."""
    kwargs = {
        "temperature": 0,
        "max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
    }
    if json_mode:
        kwargs["response_mime_type"] = "application/json"

    if genai_types is not None:
        return genai_types.GenerateContentConfig(**kwargs)
    return kwargs


def response_to_diagnostics(response):
    """Return compact diagnostics for empty or failed Gemini judge responses."""
    diagnostics = {}
    usage = getattr(response, "usage_metadata", None)
    if usage is not None:
        try:
            diagnostics["usage_metadata"] = usage.model_dump()
        except Exception:
            diagnostics["usage_metadata"] = str(usage)

    candidates = getattr(response, "candidates", None) or []
    diagnostics["n_candidates"] = len(candidates)
    diagnostics["candidates"] = []
    for cand in candidates:
        cinfo = {}
        for attr in ["finish_reason", "finish_message", "safety_ratings"]:
            val = getattr(cand, attr, None)
            if val is not None:
                try:
                    cinfo[attr] = val.model_dump()
                except Exception:
                    cinfo[attr] = str(val)
        content = getattr(cand, "content", None)
        if content is not None:
            parts = getattr(content, "parts", None) or []
            cinfo["n_parts"] = len(parts)
            cinfo["parts_preview"] = []
            for part in parts[:3]:
                text = getattr(part, "text", None)
                cinfo["parts_preview"].append((text or "")[:300])
        diagnostics["candidates"].append(cinfo)
    return diagnostics


def call_judge(judge_client, prompt):
    """Call Gemini judge. Retry once without JSON mode if the response text is empty."""
    response = judge_client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=build_judge_config(json_mode=True),
    )
    text = response.text or ""
    diagnostics = response_to_diagnostics(response)
    if text.strip() or not JUDGE_RETRY_PLAIN_ON_EMPTY:
        return text, diagnostics, "json_mode"

    retry_response = judge_client.models.generate_content(
        model=JUDGE_MODEL,
        contents=prompt,
        config=build_judge_config(json_mode=False),
    )
    retry_text = retry_response.text or ""
    retry_diagnostics = response_to_diagnostics(retry_response)
    return retry_text, {"first_attempt": diagnostics, "retry_attempt": retry_diagnostics}, "plain_retry"


def truncate_doc_text(text, max_chars):
    """Limit each retrieved document in the judge prompt to control prompt length."""
    text = text or ""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[TRUNCATED]"


# ─────────────────────────────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────────────────────────────

def mean(values):
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if not vals:
        return None
    return sum(vals) / len(vals)


def stddev(values):
    vals = [v for v in values if v is not None and not pd.isna(v)]
    if len(vals) < 2:
        return 0.0
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


# ─────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(QUERY_SUBSET_PATH):
        raise SystemExit(
            f"Missing {QUERY_SUBSET_PATH}; run evaluation/scripts/stratify_queries.py first."
        )

    queries_df = pd.read_csv(QUERY_SUBSET_PATH)
    required_cols = {"query_id", "query", "stratum"}
    missing = required_cols - set(queries_df.columns)
    if missing:
        raise SystemExit(f"{QUERY_SUBSET_PATH} is missing columns: {sorted(missing)}")

    queries_df["query_id"] = queries_df["query_id"].astype(str)
    print(f"Loaded {len(queries_df)} queries from {QUERY_SUBSET_PATH}")
    print(f"Strata: {queries_df['stratum'].value_counts().to_dict()}")

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set; cannot run LLM-Judge evaluation.")
    judge_client = genai.Client(api_key=api_key)

    run_start = datetime.now(timezone.utc)
    print(f"\nRun start: {run_start.isoformat()}")
    print(f"RAG API:    {RAG_API_URL}")
    print(f"Generator:  {GENERATOR_MODEL}")
    print(f"Judge:      {JUDGE_MODEL}")
    print(f"Configs:    {CONFIGS}")
    print(f"Top-K:      {TOP_K}")
    print(f"Judge max output tokens: {JUDGE_MAX_OUTPUT_TOKENS}")
    print(f"Judge max context chars/doc: {JUDGE_MAX_CONTEXT_CHARS_PER_DOC}")
    print(f"Retry plain on empty judge response: {JUDGE_RETRY_PLAIN_ON_EMPTY}")
    print(f"Repetitions per query/config: {N_REPETITIONS}")

    raw_records = []
    n_judge_errors = 0
    n_query_errors = 0
    n_reload_errors = 0

    for retriever, reranker in CONFIGS:
        config_label = f"{retriever}+{reranker}"
        print(f"\n{'='*72}")
        print(f"Configuration: {config_label}")
        print(f"{'='*72}")

        try:
            reload_info = reload_dataset(retriever, reranker, TOP_K)
            print(f"  Reload OK: {reload_info}")
        except Exception as e:
            print(f"  Reload failed: {e}")
            n_reload_errors += 1
            continue

        try:
            _ = query_platform("What are the tax implications of investments?", TOP_K)
            print("  Warmup OK")
        except Exception as e:
            print(f"  Warmup failed: {e}")

        for q_idx, row in queries_df.iterrows():
            query_id = str(row["query_id"])
            question = str(row["query"])
            stratum = str(row["stratum"])

            try:
                t0 = time.perf_counter()
                qresult = query_platform(question, TOP_K)
                gen_ms = (time.perf_counter() - t0) * 1000
            except Exception as e:
                print(f"  [{q_idx+1}/{len(queries_df)}] qid={query_id} GENERATE FAILED: {e}")
                n_query_errors += 1
                continue

            answer = (qresult.get("answer") or "").strip()
            retrieved_docs = qresult.get("documents", []) or []
            context_text = "\n\n---\n\n".join(
                f"[doc {i+1}] " + truncate_doc_text(d.get("text", ""), JUDGE_MAX_CONTEXT_CHARS_PER_DOC)
                for i, d in enumerate(retrieved_docs[:TOP_K])
                if isinstance(d, dict)
            )

            judge_prompt = JUDGE_PROMPT_TEMPLATE.format(
                question=question,
                context=context_text,
                answer=answer,
            )

            for rep in range(N_REPETITIONS):
                t0 = time.perf_counter()
                judge_text = None
                judge_error = None
                judge_diagnostics = None
                judge_mode = None
                parsed = None
                try:
                    judge_text, judge_diagnostics, judge_mode = call_judge(judge_client, judge_prompt)
                    parsed = parse_judge_response(judge_text)
                    if parsed is None:
                        judge_error = "empty_response" if not judge_text.strip() else "parse_failed"
                        n_judge_errors += 1
                except Exception as e:
                    judge_error = f"api_error: {type(e).__name__}: {e}"
                    n_judge_errors += 1

                judge_ms = (time.perf_counter() - t0) * 1000

                record = {
                    "config": config_label,
                    "retriever": retriever,
                    "reranker": reranker,
                    "query_id": query_id,
                    "query": question,
                    "stratum": stratum,
                    "repetition": rep,
                    "answer": answer,
                    "n_context_docs": len(retrieved_docs),
                    "generation_ms": round(gen_ms, 2),
                    "judge_ms": round(judge_ms, 2),
                    "judge_prompt": judge_prompt,
                    "judge_raw_response": judge_text,
                    "judge_error": judge_error,
                    "judge_mode": judge_mode,
                    "judge_diagnostics": judge_diagnostics,
                    "groundedness": parsed["groundedness"] if parsed else None,
                    "relevance": parsed["relevance"] if parsed else None,
                    "completeness": parsed["completeness"] if parsed else None,
                    "groundedness_reason": parsed["groundedness_reason"] if parsed else None,
                    "relevance_reason": parsed["relevance_reason"] if parsed else None,
                    "completeness_reason": parsed["completeness_reason"] if parsed else None,
                }
                raw_records.append(record)

            recent_records = raw_records[-N_REPETITIONS:]
            scores_summary = []
            for r in recent_records:
                if r["groundedness"] is not None:
                    scores_summary.append(
                        f"G={r['groundedness']}/R={r['relevance']}/C={r['completeness']}"
                    )
                else:
                    scores_summary.append("FAIL")
            print(
                f"  [{q_idx+1}/{len(queries_df)}] qid={query_id:<8} {stratum:<6} → "
                f"{' | '.join(scores_summary)}"
            )

    if not raw_records:
        raise RuntimeError(
            "No judge records were produced. Check RAG_API_URL, /reload-dataset, "
            "GEMINI_API_KEY, and whether the platform is running."
        )

    print(f"\nWriting raw judge records → {RAW_OUT}")
    with open(RAW_OUT, "w", encoding="utf-8") as f:
        for r in raw_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    raw_df = pd.DataFrame(raw_records)

    print(f"Computing per-query summary → {SUMMARY_OUT}")
    summary_rows = []
    for (config, query_id), group in raw_df.groupby(["config", "query_id"]):
        first = group.iloc[0]
        summary_rows.append({
            "config": config,
            "query_id": query_id,
            "stratum": first["stratum"],
            "groundedness_mean": mean(group["groundedness"].tolist()),
            "groundedness_std":  stddev(group["groundedness"].tolist()),
            "relevance_mean":    mean(group["relevance"].tolist()),
            "relevance_std":     stddev(group["relevance"].tolist()),
            "completeness_mean": mean(group["completeness"].tolist()),
            "completeness_std":  stddev(group["completeness"].tolist()),
            "n_judge_passes":    int(group["groundedness"].notna().sum()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(SUMMARY_OUT, index=False, float_format="%.4f")

    print(f"Computing aggregate summary → {AGGREGATE_OUT}")
    aggregate_rows = []
    for config in summary_df["config"].dropna().unique():
        sub = summary_df[summary_df["config"] == config]
        for stratum in ["easy", "medium", "hard", "all"]:
            scope = sub if stratum == "all" else sub[sub["stratum"] == stratum]
            if scope.empty:
                continue
            aggregate_rows.append({
                "config": config,
                "stratum": stratum,
                "n_queries": len(scope),
                "groundedness_mean": scope["groundedness_mean"].mean(),
                "relevance_mean":    scope["relevance_mean"].mean(),
                "completeness_mean": scope["completeness_mean"].mean(),
                "groundedness_overall_std": scope["groundedness_mean"].std() if len(scope) > 1 else 0,
                "relevance_overall_std":    scope["relevance_mean"].std()    if len(scope) > 1 else 0,
                "completeness_overall_std": scope["completeness_mean"].std() if len(scope) > 1 else 0,
            })
    aggregate_df = pd.DataFrame(aggregate_rows)
    aggregate_df.to_csv(AGGREGATE_OUT, index=False, float_format="%.4f")

    metadata = {
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": datetime.now(timezone.utc).isoformat(),
        "rag_api_url": RAG_API_URL,
        "generator_model": GENERATOR_MODEL,
        "judge_model": JUDGE_MODEL,
        "n_queries": int(len(queries_df)),
        "configs_evaluated": [list(c) for c in CONFIGS],
        "n_repetitions_per_query_config": N_REPETITIONS,
        "top_k": TOP_K,
        "judge_max_output_tokens": JUDGE_MAX_OUTPUT_TOKENS,
        "judge_max_context_chars_per_doc": JUDGE_MAX_CONTEXT_CHARS_PER_DOC,
        "judge_retry_plain_on_empty": JUDGE_RETRY_PLAIN_ON_EMPTY,
        "rubric": ["groundedness", "relevance", "completeness"],
        "rubric_scale": "1-5 ordinal",
        "n_judge_records_total": len(raw_records),
        "n_judge_errors": n_judge_errors,
        "n_query_errors": n_query_errors,
        "n_reload_errors": n_reload_errors,
        "stratum_sample_sizes": queries_df["stratum"].value_counts().to_dict(),
    }
    with open(METADATA_OUT, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*72}")
    print("HEADLINE — judge scores by configuration")
    print(f"{'='*72}")
    print(f"{'Config':<26} {'Stratum':<8} {'N':>4} {'Ground.':>8} {'Rel.':>8} {'Compl.':>8}")
    print("-" * 72)
    for _, r in aggregate_df.iterrows():
        print(f"{r['config']:<26} {r['stratum']:<8} {int(r['n_queries']):>4} "
              f"{r['groundedness_mean']:>8.3f} {r['relevance_mean']:>8.3f} "
              f"{r['completeness_mean']:>8.3f}")

    print(f"\nTotal judge records: {len(raw_records)}")
    print(f"Judge errors: {n_judge_errors}")
    print(f"Query/generate errors: {n_query_errors}")
    print(f"Reload errors: {n_reload_errors}")


if __name__ == "__main__":
    main()
