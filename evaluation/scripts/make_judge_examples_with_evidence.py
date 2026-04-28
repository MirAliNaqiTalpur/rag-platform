import json
import re
import csv
from pathlib import Path

RAW_PATH = Path("evaluation/results/judge_raw_responses.jsonl")
TXT_OUT = Path("evaluation/results/judge_examples_with_evidence.txt")
CSV_OUT = Path("evaluation/results/judge_examples_with_evidence.csv")

MAX_EVIDENCE_CHARS = 900
MAX_ANSWER_CHARS = 900
MAX_REASON_CHARS = 350


def extract_context_from_prompt(prompt: str) -> str:
    """Extract the retrieved context block from the judge prompt."""
    if not prompt:
        return ""

    start_marker = "RETRIEVED CONTEXT:"
    end_marker = "GENERATED ANSWER:"

    start = prompt.find(start_marker)
    end = prompt.find(end_marker)

    if start == -1 or end == -1 or end <= start:
        return ""

    context = prompt[start + len(start_marker):end].strip()
    return context


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text


def shorten(text: str, max_chars: int) -> str:
    text = clean_text(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "..."


def make_evidence_excerpt(prompt: str) -> str:
    context = extract_context_from_prompt(prompt)

    if not context:
        return ""

    # Split by document separator used in llm_judge.py
    docs = re.split(r"\n\s*---\s*\n", context)

    excerpts = []
    for i, doc in enumerate(docs[:3], start=1):
        doc = clean_text(doc)
        if not doc:
            continue

        # Keep each doc short
        excerpts.append(f"Doc {i}: {shorten(doc, 280)}")

    return "\n".join(excerpts)


def main():
    if not RAW_PATH.exists():
        raise SystemExit(f"Missing file: {RAW_PATH}")

    records = [
        json.loads(line)
        for line in RAW_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    rows = []
    text_lines = []

    for r in records:
        evidence_excerpt = make_evidence_excerpt(r.get("judge_prompt", ""))

        row = {
            "config": r.get("config"),
            "query_id": r.get("query_id"),
            "stratum": r.get("stratum"),
            "query": r.get("query"),
            "evidence_excerpt": evidence_excerpt,
            "answer_excerpt": shorten(r.get("answer", ""), MAX_ANSWER_CHARS),
            "groundedness": r.get("groundedness"),
            "relevance": r.get("relevance"),
            "completeness": r.get("completeness"),
            "groundedness_reason": shorten(r.get("groundedness_reason", ""), MAX_REASON_CHARS),
            "relevance_reason": shorten(r.get("relevance_reason", ""), MAX_REASON_CHARS),
            "completeness_reason": shorten(r.get("completeness_reason", ""), MAX_REASON_CHARS),
            "judge_error": r.get("judge_error"),
        }
        rows.append(row)

        text_lines.extend([
            "=" * 100,
            f"CONFIG: {row['config']}",
            f"QUERY ID: {row['query_id']}",
            f"STRATUM: {row['stratum']}",
            f"QUERY: {row['query']}",
            "",
            "RETRIEVED EVIDENCE EXCERPT:",
            row["evidence_excerpt"] or "[No evidence excerpt found]",
            "",
            "GENERATED ANSWER EXCERPT:",
            row["answer_excerpt"],
            "",
            f"JUDGE SCORES: Groundedness={row['groundedness']} ; Relevance={row['relevance']} ; Completeness={row['completeness']}",
            "",
            "JUDGE REASONS:",
            f"- Groundedness: {row['groundedness_reason']}",
            f"- Relevance: {row['relevance_reason']}",
            f"- Completeness: {row['completeness_reason']}",
            "",
        ])

    TXT_OUT.write_text("\n".join(text_lines), encoding="utf-8")

    with CSV_OUT.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved text review: {TXT_OUT}")
    print(f"Saved CSV review:  {CSV_OUT}")


if __name__ == "__main__":
    main()
