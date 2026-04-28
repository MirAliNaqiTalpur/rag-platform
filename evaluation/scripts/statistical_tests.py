"""
Statistical significance testing for the retrieval evaluation.

For a 9-configuration ablation, we run paired Wilcoxon signed-rank tests
on per-query metrics (MRR@10, nDCG@10) across all C(9,2)=36 pairwise
configuration comparisons.

Why paired Wilcoxon?
  - The same query set is run through every configuration → naturally
    paired observations.
  - Wilcoxon makes no parametric assumptions about the distribution
    of metric differences. IR metrics are bounded [0, 1] and often
    non-normally distributed → t-test assumptions are violated.
  - Standard practice in IR evaluation literature (e.g., Smucker et al.
    2007 "A Comparison of Statistical Significance Tests for Information
    Retrieval Evaluation").

Why Bonferroni correction?
  - With 36 comparisons at α=0.05 raw, we'd expect ~1.8 false positives
    by chance alone. Bonferroni controls family-wise error rate by
    requiring p < α/m where m is the number of comparisons.
  - Conservative (some say overly so), but defensible in any committee.
    The alternative — Benjamini-Hochberg FDR — is reported as a
    secondary table for completeness.

Outputs:
  - retrieval_pairwise_tests.csv : full table of 36 pairwise tests
                                   with raw p, Bonferroni p, BH p,
                                   effect size, and direction
  - retrieval_significance_summary.txt : human-readable summary
                                         organized by which configs
                                         significantly differ

Usage:
    python evaluation/scripts/statistical_tests.py
"""

import os
from itertools import combinations
from pathlib import Path

import pandas as pd
from scipy import stats


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = "evaluation/results"
DETAILED_PATH = os.path.join(RESULTS_DIR, "retrieval_detailed_results.csv")
PAIRWISE_OUT = os.path.join(RESULTS_DIR, "retrieval_pairwise_tests.csv")
SUMMARY_OUT = os.path.join(RESULTS_DIR, "retrieval_significance_summary.txt")

# Significance threshold (family-wise α before correction)
ALPHA = 0.05

# K value to test on. K=10 is the standard choice — Recall@10 is the
# usual headline retrieval metric and gives the discriminator the most
# room to differentiate configs.
TEST_K = 10

# Metrics to test. nDCG@10 is the primary; MRR@10 is the secondary.
METRICS = ["ndcg_at_k", "mrr_at_k"]


# ─────────────────────────────────────────────────────────────────────
# Effect size — Cliff's delta (non-parametric, paired-data appropriate)
# ─────────────────────────────────────────────────────────────────────

def cliffs_delta(x, y):
    """Cliff's delta: probability(x > y) - probability(x < y).

    Range: [-1, +1]
        +1 means every x > every y
        -1 means every x < every y
         0 means equal distributions

    Conventional interpretation (Romano et al. 2006):
        |d| < 0.147 → negligible
        |d| < 0.33  → small
        |d| < 0.474 → medium
        |d| >= 0.474 → large
    """
    n_gt = sum(1 for xi in x for yi in y if xi > yi)
    n_lt = sum(1 for xi in x for yi in y if xi < yi)
    n_total = len(x) * len(y)
    if n_total == 0:
        return 0.0
    return (n_gt - n_lt) / n_total


def magnitude_label(d):
    abs_d = abs(d)
    if abs_d < 0.147:
        return "negligible"
    if abs_d < 0.33:
        return "small"
    if abs_d < 0.474:
        return "medium"
    return "large"


# ─────────────────────────────────────────────────────────────────────
# Bonferroni & Benjamini-Hochberg corrections
# ─────────────────────────────────────────────────────────────────────

def bh_adjust(p_values):
    """Benjamini-Hochberg adjusted p-values (controls FDR rather than FWER)."""
    n = len(p_values)
    if n == 0:
        return []
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    prev = 1.0
    # Walk in descending order of p (largest first) to enforce monotonicity
    for rank in range(n, 0, -1):
        original_idx, p = indexed[rank - 1]
        adj = p * n / rank
        if adj > prev:
            adj = prev
        else:
            prev = adj
        adjusted[original_idx] = min(1.0, adj)
    return adjusted


# ─────────────────────────────────────────────────────────────────────
# Significance label
# ─────────────────────────────────────────────────────────────────────

def stars(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "ns"


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    detailed = pd.read_csv(DETAILED_PATH)

    # Filter to test K
    df = detailed[detailed["k"] == TEST_K].copy()
    df["config"] = df["retriever"] + "+" + df["reranker"]

    configs = sorted(df["config"].unique())
    print(f"Found {len(configs)} configurations")
    print(f"Configs: {configs}")
    print(f"K = {TEST_K}, n_queries = {df['query_id'].nunique()}")
    print()

    n_comparisons = len(configs) * (len(configs) - 1) // 2
    print(f"Pairwise comparisons: {n_comparisons}")
    print(f"Bonferroni-corrected α: {ALPHA}/{n_comparisons} = {ALPHA/n_comparisons:.5f}")
    print()

    rows = []

    for metric in METRICS:
        # Pivot: query_id × config → metric value
        pivot = df.pivot_table(
            index="query_id",
            columns="config",
            values=metric,
            aggfunc="mean",
        )

        # Drop queries that don't have all configs (shouldn't happen, but defensive)
        pivot = pivot.dropna()
        if len(pivot) == 0:
            print(f"⚠ No paired queries for {metric}")
            continue

        n = len(pivot)
        print(f"\nMetric: {metric}, paired queries: {n}")

        for cfg_a, cfg_b in combinations(configs, 2):
            x = pivot[cfg_a].values
            y = pivot[cfg_b].values

            # Wilcoxon signed-rank test.
            # If all paired diffs are zero, scipy raises; treat as p=1.
            diffs = x - y
            if (diffs == 0).all():
                stat, p = float("nan"), 1.0
            else:
                # zero_method='wilcox' is the standard (drops zero diffs).
                # alternative='two-sided' is the default and what we want.
                try:
                    stat, p = stats.wilcoxon(x, y, zero_method="wilcox")
                except ValueError:
                    # Tiny samples or pathological inputs
                    stat, p = float("nan"), 1.0

            d = cliffs_delta(x, y)
            mean_diff = float((x - y).mean())

            rows.append({
                "metric": metric,
                "config_a": cfg_a,
                "config_b": cfg_b,
                "mean_a": float(x.mean()),
                "mean_b": float(y.mean()),
                "mean_diff": mean_diff,
                "wilcoxon_stat": float(stat) if stat == stat else None,
                "p_raw": float(p),
                "cliffs_delta": float(d),
                "effect_magnitude": magnitude_label(d),
                "n_queries": n,
            })

    # Build results df, add corrections per metric
    results = pd.DataFrame(rows)

    # Add Bonferroni and BH per metric
    parts = []
    for metric, group in results.groupby("metric"):
        g = group.copy()
        m = len(g)
        g["p_bonferroni"] = (g["p_raw"] * m).clip(upper=1.0)
        g["p_bh"] = bh_adjust(g["p_raw"].tolist())
        g["significant_raw"]    = g["p_raw"]         < ALPHA
        g["significant_bonf"]   = g["p_bonferroni"]  < ALPHA
        g["significant_bh"]     = g["p_bh"]          < ALPHA
        g["sig_label"]          = g["p_bonferroni"].apply(stars)
        parts.append(g)
    results = pd.concat(parts, ignore_index=True)

    # Order columns nicely
    col_order = [
        "metric", "config_a", "config_b",
        "mean_a", "mean_b", "mean_diff",
        "wilcoxon_stat", "p_raw", "p_bonferroni", "p_bh",
        "significant_raw", "significant_bonf", "significant_bh",
        "sig_label",
        "cliffs_delta", "effect_magnitude",
        "n_queries",
    ]
    results = results[col_order]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    results.to_csv(PAIRWISE_OUT, index=False, float_format="%.6f")
    print(f"\nSaved pairwise results: {PAIRWISE_OUT}")

    # ─────────────────────────────────────────────────────────
    # Human-readable summary
    # ─────────────────────────────────────────────────────────
    lines = []
    lines.append("=" * 80)
    lines.append("STATISTICAL SIGNIFICANCE SUMMARY")
    lines.append("=" * 80)
    lines.append(f"Test:        Paired Wilcoxon signed-rank, two-sided")
    lines.append(f"K:           {TEST_K}")
    lines.append(f"Queries:     {df['query_id'].nunique()} (paired across all configs)")
    lines.append(f"Comparisons: {n_comparisons} per metric")
    lines.append(f"Correction:  Bonferroni (α/m = {ALPHA/n_comparisons:.5f})")
    lines.append(f"Effect size: Cliff's delta (non-parametric)")
    lines.append("")
    lines.append("Significance labels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    lines.append("(All p-values are Bonferroni-corrected.)")

    for metric in METRICS:
        m_results = results[results["metric"] == metric].copy()
        m_results = m_results.sort_values("p_bonferroni")

        lines.append("")
        lines.append("=" * 80)
        lines.append(f"METRIC: {metric}")
        lines.append("=" * 80)

        # Significant after Bonferroni
        sig = m_results[m_results["significant_bonf"]]
        notsig = m_results[~m_results["significant_bonf"]]

        lines.append("")
        lines.append(f"-- Significantly different after Bonferroni ({len(sig)} of {len(m_results)} pairs) --")
        lines.append("")
        if len(sig) == 0:
            lines.append("  (none)")
        else:
            lines.append(f"  {'Config A':<26} {'Config B':<26} "
                         f"{'Δ mean':>9} {'p-Bonf':>10} {'sig':>5} {'effect':>13}")
            lines.append("  " + "-" * 90)
            for _, r in sig.iterrows():
                # Direction: positive means A > B
                lines.append(f"  {r['config_a']:<26} {r['config_b']:<26} "
                             f"{r['mean_diff']:>+9.4f} {r['p_bonferroni']:>10.5f} "
                             f"{r['sig_label']:>5} {r['effect_magnitude']:>13}")

        lines.append("")
        lines.append(f"-- Not significantly different ({len(notsig)} of {len(m_results)} pairs) --")
        lines.append("")
        if len(notsig) == 0:
            lines.append("  (none)")
        else:
            lines.append(f"  {'Config A':<26} {'Config B':<26} "
                         f"{'Δ mean':>9} {'p-Bonf':>10}")
            lines.append("  " + "-" * 80)
            for _, r in notsig.iterrows():
                lines.append(f"  {r['config_a']:<26} {r['config_b']:<26} "
                             f"{r['mean_diff']:>+9.4f} {r['p_bonferroni']:>10.5f}")

    summary_text = "\n".join(lines) + "\n"
    with open(SUMMARY_OUT, "w", encoding="utf-8") as f:
        f.write(summary_text)
    print(f"Saved human-readable summary: {SUMMARY_OUT}")

    # Print to stdout too
    print()
    print(summary_text)


if __name__ == "__main__":
    main()
