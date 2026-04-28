import json
from pathlib import Path

raw_path = Path("evaluation/results/judge_raw_responses.jsonl")
out_path = Path("evaluation/results/judge_readable_review.txt")

if not raw_path.exists():
    raise SystemExit(f"Missing file: {raw_path}")

lines = []

for r in map(json.loads, raw_path.read_text(encoding="utf-8").splitlines()):
    scores = (
        f"Groundedness={r.get('groundedness')} ; "
        f"Relevance={r.get('relevance')} ; "
        f"Completeness={r.get('completeness')}"
    )

    lines.extend([
        "=" * 100,
        f"CONFIG: {r.get('config')}",
        f"QUERY ID: {r.get('query_id')}",
        f"STRATUM: {r.get('stratum')}",
        f"QUERY: {r.get('query')}",
        f"SCORES: {scores}",
        "",
        f"GROUNDING REASON: {r.get('groundedness_reason')}",
        f"RELEVANCE REASON: {r.get('relevance_reason')}",
        f"COMPLETENESS REASON: {r.get('completeness_reason')}",
        "",
        "GENERATED ANSWER:",
        r.get("answer", ""),
        "",
    ])

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved readable review to: {out_path}")
