"""
Generate publication-quality figures for Chapter 6.

Three figures:

  1. retrieval_heatmap_ndcg.png
     3×3 heatmap (retriever × reranker) of mean nDCG@10 with text labels.
     Uses a sequential colormap (viridis) with values printed in cells.
     This is the headline ablation figure.

  2. retrieval_latency_quality.png
     Scatter of median backend latency (x, log scale) vs nDCG@10 (y).
     Each of the 9 configs is a labeled point. Pareto frontier shaded.
     This is the engineering tradeoff story.

  3. retrieval_metrics_by_k.png
     Six small multiples (3 retrievers × 2 metrics: Recall, MRR) showing
     how the metric evolves K=3 → K=5 → K=10 for each reranker variant.
     This shows that conclusions are stable across K choices.

All figures save at 300 DPI for publication. Style is restrained
(black-and-white-friendly, no gridlines, no chartjunk).

Usage:
    python evaluation/scripts/generate_figures.py
"""

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

RESULTS_DIR = "evaluation/results"
SUMMARY_PATH = os.path.join(RESULTS_DIR, "retrieval_summary_results.csv")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Display order for the 3×3 grid — preserves a logical progression
RETRIEVER_ORDER = ["bm25", "simple", "hybrid"]
RERANKER_ORDER = ["none", "simple", "cross_encoder"]

# Display labels (cleaner than raw config names)
RETRIEVER_LABELS = {
    "bm25":   "BM25",
    "simple": "Dense",
    "hybrid": "Hybrid (RRF)",
}
RERANKER_LABELS = {
    "none":          "No reranker",
    "simple":        "Lexical rerank",
    "cross_encoder": "Cross-encoder",
}

# Consistent colors for retrievers across all figures
RETRIEVER_COLORS = {
    "bm25":   "#d97706",  # amber
    "simple": "#1d4ed8",  # blue
    "hybrid": "#059669",  # green
}

# Reranker marker shapes
RERANKER_MARKERS = {
    "none":          "o",
    "simple":        "s",
    "cross_encoder": "^",
}


def style():
    """Apply a clean, restrained matplotlib style."""
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "legend.frameon": False,
    })


# ─────────────────────────────────────────────────────────────────────
# Figure 1 — nDCG@10 heatmap
# ─────────────────────────────────────────────────────────────────────

def figure_heatmap(df: pd.DataFrame):
    """3×3 heatmap of nDCG@10 across retriever × reranker."""
    df = df[df["k"] == 10].copy()

    matrix = np.zeros((len(RETRIEVER_ORDER), len(RERANKER_ORDER)))
    for i, retr in enumerate(RETRIEVER_ORDER):
        for j, rer in enumerate(RERANKER_ORDER):
            row = df[(df["retriever"] == retr) & (df["reranker"] == rer)]
            if not row.empty:
                matrix[i, j] = float(row["ndcg_at_k"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Use a sequential colormap; nDCG is bounded [0, 1] so we fix vmin/vmax
    # to the observed range with a small margin
    vmin = max(0, matrix.min() - 0.05)
    vmax = min(1, matrix.max() + 0.05)
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    # Axis labels
    ax.set_xticks(range(len(RERANKER_ORDER)))
    ax.set_xticklabels([RERANKER_LABELS[r] for r in RERANKER_ORDER])
    ax.set_yticks(range(len(RETRIEVER_ORDER)))
    ax.set_yticklabels([RETRIEVER_LABELS[r] for r in RETRIEVER_ORDER])

    ax.set_xlabel("Reranker")
    ax.set_ylabel("Retriever")
    ax.set_title("nDCG@10 across the 3 × 3 ablation")

    # Cell value annotations
    for i in range(len(RETRIEVER_ORDER)):
        for j in range(len(RERANKER_ORDER)):
            v = matrix[i, j]
            # Choose text color for readability against background
            text_color = "white" if v < (vmin + vmax) / 2 else "black"
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    color=text_color, fontsize=12, fontweight="bold")

    # Color bar
    cbar = plt.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    cbar.set_label("nDCG@10", rotation=270, labelpad=15)

    out = os.path.join(FIGURES_DIR, "retrieval_heatmap_ndcg.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────
# Figure 2 — Latency vs Quality scatter
# ─────────────────────────────────────────────────────────────────────

def figure_latency_quality(df: pd.DataFrame):
    """Scatter: median backend latency (log-x) vs nDCG@10 (y)."""
    df = df[df["k"] == 10].copy()

    fig, ax = plt.subplots(figsize=(8, 5))

    for _, row in df.iterrows():
        retr, rer = row["retriever"], row["reranker"]
        x = float(row["backend_total_ms_median"])
        y = float(row["ndcg_at_k"])

        ax.scatter(
            x, y,
            s=180,
            color=RETRIEVER_COLORS[retr],
            marker=RERANKER_MARKERS[rer],
            edgecolor="black",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )

        # Compact label near each point
        ax.annotate(
            f"{retr}+{rer}",
            xy=(x, y),
            xytext=(8, 4),
            textcoords="offset points",
            fontsize=8,
            color="#444",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Median backend latency (ms, log scale)")
    ax.set_ylabel("nDCG@10")
    ax.set_title("Quality vs latency tradeoff across the 9 configurations")

    # Two-row legend: retriever colors + reranker markers
    from matplotlib.lines import Line2D
    legend_elements = []
    legend_elements.append(Line2D([], [], color="white", label="Retriever:"))
    for retr in RETRIEVER_ORDER:
        legend_elements.append(Line2D([], [], marker="s", linestyle="",
            markerfacecolor=RETRIEVER_COLORS[retr],
            markeredgecolor="black",
            markersize=8,
            label=f"  {RETRIEVER_LABELS[retr]}"))
    legend_elements.append(Line2D([], [], color="white", label="Reranker:"))
    for rer in RERANKER_ORDER:
        legend_elements.append(Line2D([], [], marker=RERANKER_MARKERS[rer],
            linestyle="",
            markerfacecolor="#888",
            markeredgecolor="black",
            markersize=8,
            label=f"  {RERANKER_LABELS[rer]}"))
    ax.legend(handles=legend_elements, loc="lower right", ncol=1)

    out = os.path.join(FIGURES_DIR, "retrieval_latency_quality.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────
# Figure 3 — Metric evolution by K
# ─────────────────────────────────────────────────────────────────────

def figure_metrics_by_k(df: pd.DataFrame):
    """6-panel grid: (3 retrievers) × (Recall, MRR), curves over K."""
    metrics = [
        ("recall_at_k", "Recall@K"),
        ("mrr_at_k", "MRR@K"),
    ]

    fig, axes = plt.subplots(
        len(metrics), len(RETRIEVER_ORDER),
        figsize=(11, 6),
        sharex=True,
        sharey="row",
    )

    for row_idx, (metric_col, metric_label) in enumerate(metrics):
        for col_idx, retr in enumerate(RETRIEVER_ORDER):
            ax = axes[row_idx, col_idx]

            for rer in RERANKER_ORDER:
                sub = df[(df["retriever"] == retr) & (df["reranker"] == rer)]
                sub = sub.sort_values("k")
                ax.plot(
                    sub["k"], sub[metric_col],
                    marker=RERANKER_MARKERS[rer],
                    color=RETRIEVER_COLORS[retr],
                    label=RERANKER_LABELS[rer] if (row_idx == 0 and col_idx == 0) else None,
                    linewidth=1.6,
                    markersize=7,
                    markeredgecolor="black",
                    markeredgewidth=0.4,
                )

            if row_idx == 0:
                ax.set_title(RETRIEVER_LABELS[retr], color=RETRIEVER_COLORS[retr])
            if col_idx == 0:
                ax.set_ylabel(metric_label)
            if row_idx == len(metrics) - 1:
                ax.set_xlabel("K")
            ax.set_xticks([3, 5, 10])

    # Single legend at top
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, "retrieval_metrics_by_k.png")
    plt.savefig(out)
    plt.close()
    print(f"  Saved {out}")


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    style()

    df = pd.read_csv(SUMMARY_PATH)
    print(f"Loaded {len(df)} summary rows from {SUMMARY_PATH}")

    print("\nGenerating figures...")
    figure_heatmap(df)
    figure_latency_quality(df)
    figure_metrics_by_k(df)
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == "__main__":
    main()
