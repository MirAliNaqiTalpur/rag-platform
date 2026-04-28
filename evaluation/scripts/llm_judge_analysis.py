"""
Analyze the LLM-as-a-Judge results and produce the Chapter 6.7 summary.

Reads judge_summary.csv produced by llm_judge.py, runs paired statistical
tests between the two configurations on each rubric, and writes a
human-readable summary file for Section 6.7.

Outputs:
  evaluation/results/judge_pairwise_tests.csv
      Wilcoxon signed-rank tests on per-query mean scores between
      simple+none and simple+cross_encoder, with Holm-corrected p-values.
  evaluation/results/judge_section_6_7_summary.txt
      Plain-text summary of findings for report writing.

Usage:
    python evaluation/scripts/llm_judge_analysis.py
"""

import os

import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = "evaluation/results"
SUMMARY_PATH = os.path.join(RESULTS_DIR, "judge_summary.csv")
AGGREGATE_PATH = os.path.join(RESULTS_DIR, "judge_aggregate.csv")

PAIRWISE_OUT = os.path.join(RESULTS_DIR, "judge_pairwise_tests.csv")
TEXT_OUT = os.path.join(RESULTS_DIR, "judge_section_6_7_summary.txt")

CONFIG_A = "simple+none"
CONFIG_B = "simple+cross_encoder"

RUBRICS = ["groundedness", "relevance", "completeness"]
ALPHA = 0.05


def cohens_dz(diffs):
    """Cohen's dz for paired samples."""
    diffs = pd.Series(diffs).dropna()
    if len(diffs) < 2:
        return 0.0
    sd = diffs.std(ddof=1)
    if sd == 0:
        return 0.0
    return float(diffs.mean() / sd)


def dz_magnitude(d):
    a = abs(d)
    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"
    return "large"


def add_holm_correction(df, p_col="p_value", alpha=0.05):
    """Add Holm-adjusted p-values and significance decisions."""
    if df.empty:
        return df

    out = df.copy()
    sorted_idx = out[p_col].sort_values(kind="mergesort").index.tolist()
    m = len(sorted_idx)
    adjusted = {}
    running_max = 0.0

    for rank, idx in enumerate(sorted_idx):
        raw_p = float(out.loc[idx, p_col])
        adj_p = min(1.0, (m - rank) * raw_p)
        running_max = max(running_max, adj_p)
        adjusted[idx] = running_max

    out["holm_p_value"] = pd.Series(adjusted)
    out["significant_holm"] = out["holm_p_value"] < alpha
    return out


def main():
    if not os.path.exists(SUMMARY_PATH):
        raise SystemExit(f"Missing {SUMMARY_PATH}; run llm_judge.py first.")

    summary = pd.read_csv(SUMMARY_PATH)
    aggregate = pd.read_csv(AGGREGATE_PATH) if os.path.exists(AGGREGATE_PATH) else None

    required_cols = {"config", "query_id"} | {f"{r}_mean" for r in RUBRICS}
    missing = required_cols - set(summary.columns)
    if missing:
        raise SystemExit(f"{SUMMARY_PATH} is missing columns: {sorted(missing)}")

    pairs = []
    for rubric in RUBRICS:
        col = f"{rubric}_mean"

        pivot = summary.pivot_table(
            index="query_id", columns="config", values=col, aggfunc="mean"
        ).dropna()

        if CONFIG_A not in pivot.columns or CONFIG_B not in pivot.columns:
            print(f"Missing one of the configs for rubric {rubric}; skipping.")
            continue

        x = pivot[CONFIG_A].astype(float).values
        y = pivot[CONFIG_B].astype(float).values
        diffs = x - y

        if len(diffs) == 0:
            continue

        if (diffs == 0).all():
            stat, p = float("nan"), 1.0
        else:
            try:
                stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
            except ValueError:
                stat, p = float("nan"), 1.0

        dz = cohens_dz(diffs)

        pairs.append({
            "rubric": rubric,
            "config_a": CONFIG_A,
            "config_b": CONFIG_B,
            "mean_a": float(x.mean()),
            "mean_b": float(y.mean()),
            "mean_diff_a_minus_b": float(diffs.mean()),
            "wilcoxon_stat": float(stat) if stat == stat else None,
            "p_value": float(p),
            "significant_uncorrected": bool(p < ALPHA),
            "cohens_dz": float(dz),
            "effect_magnitude": dz_magnitude(dz),
            "n_queries": int(len(x)),
        })

    pairs_df = pd.DataFrame(pairs)
    if pairs_df.empty:
        raise RuntimeError(
            "No pairwise tests were computed. Check that judge_summary.csv contains "
            "both simple+none and simple+cross_encoder for the same query_ids."
        )

    pairs_df = add_holm_correction(pairs_df, alpha=ALPHA)
    pairs_df.to_csv(PAIRWISE_OUT, index=False, float_format="%.6f")
    print(f"Saved pairwise tests → {PAIRWISE_OUT}")

    lines = []
    lines.append("=" * 76)
    lines.append("LLM-AS-A-JUDGE SUMMARY — for Section 6.7")
    lines.append("=" * 76)
    lines.append("")
    lines.append("Setup:")
    lines.append("  - Generator: configured platform Gemini model used by /query")
    lines.append("  - Judge:     Gemini 2.5 Pro at temperature 0")
    lines.append("  - Rubric:    Groundedness, Relevance, Completeness on 1-5 ordinal scale")
    lines.append("  - Configs:   simple+none vs simple+cross_encoder")
    lines.append("  - Queries:   stratified FiQA subset")
    lines.append("  - Reps:      independent judge passes per query/config")
    lines.append("")
    lines.append("Statistical test: paired Wilcoxon signed-rank on per-query mean scores.")
    lines.append(f"Significance threshold: α = {ALPHA}, with Holm correction across rubrics.")
    lines.append("Effect size: Cohen's dz on paired score differences.")
    lines.append("")

    if aggregate is not None:
        lines.append("─" * 76)
        lines.append("AGGREGATE SCORES")
        lines.append("─" * 76)
        lines.append("")
        lines.append(f"{'Config':<26} {'Stratum':<8} {'N':>4} {'Ground.':>9} {'Rel.':>9} {'Compl.':>9}")
        lines.append("-" * 76)
        for _, r in aggregate.iterrows():
            lines.append(
                f"{r['config']:<26} {r['stratum']:<8} {int(r['n_queries']):>4} "
                f"{r['groundedness_mean']:>9.3f} {r['relevance_mean']:>9.3f} "
                f"{r['completeness_mean']:>9.3f}"
            )
        lines.append("")

    lines.append("─" * 76)
    lines.append("PAIRWISE TESTS (cross_encoder vs none, on dense retrieval)")
    lines.append("─" * 76)
    lines.append("")
    for _, r in pairs_df.iterrows():
        sig = "YES" if r["significant_holm"] else "no"
        direction = (
            "cross_encoder higher" if r["mean_diff_a_minus_b"] < 0
            else "none higher" if r["mean_diff_a_minus_b"] > 0
            else "tie"
        )
        lines.append(f"Rubric: {r['rubric']}")
        lines.append(f"  none mean:              {r['mean_a']:.3f}")
        lines.append(f"  cross_encoder mean:     {r['mean_b']:.3f}")
        lines.append(f"  Δ (none − xenc):        {r['mean_diff_a_minus_b']:+.3f}  ({direction})")
        lines.append(f"  Wilcoxon p-value:       {r['p_value']:.4f}")
        lines.append(f"  Holm-adjusted p-value:  {r['holm_p_value']:.4f}")
        lines.append(f"  Significant after Holm: {sig}")
        lines.append(f"  Cohen's dz:             {r['cohens_dz']:+.3f}  ({r['effect_magnitude']})")
        lines.append("")

    lines.append("─" * 76)
    lines.append("INTERPRETATION FOR SECTION 6.7")
    lines.append("─" * 76)
    lines.append("")

    n_significant = int(pairs_df["significant_holm"].sum())
    if n_significant == 0:
        lines.append("Headline: cross-encoder reranking does not produce Holm-corrected")
        lines.append("statistically significant differences in any of the three judge-assessed")
        lines.append("rubrics on the dense retriever. This is consistent with the retrieval-stage")
        lines.append("finding that cross-encoder reranking was statistically neutral on dense")
        lines.append("retrieval, and extends that conclusion to downstream answer quality under")
        lines.append("the tested setup.")
    else:
        lines.append(f"Headline: {n_significant} of {len(pairs_df)} rubrics show a Holm-corrected")
        lines.append("statistically significant difference between cross-encoder and no reranking")
        lines.append("on the dense retriever. Specific rubrics:")
        for _, r in pairs_df[pairs_df["significant_holm"]].iterrows():
            direction = "cross_encoder" if r["mean_diff_a_minus_b"] < 0 else "no reranking"
            lines.append(
                f"  - {r['rubric']}: {direction} scores higher "
                f"(absolute Δ = {abs(r['mean_diff_a_minus_b']):.3f}, "
                f"Holm p = {r['holm_p_value']:.4f})"
            )
    lines.append("")

    text = "\n".join(lines) + "\n"
    with open(TEXT_OUT, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Saved summary → {TEXT_OUT}")
    print()
    print(text)


if __name__ == "__main__":
    main()
