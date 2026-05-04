
import pandas as pd
from pathlib import Path

src = Path("evaluation/results/retrieval_pairwise_tests.csv")
if not src.exists():
    src = Path("retrieval_pairwise_tests.csv")

df = pd.read_csv(src)

def clean_config(x):
    parts = str(x).split("+")
    if len(parts) == 2:
        retriever, reranker = parts
        if retriever == "simple":
            retriever = "dense"
        return f"{retriever}+{reranker}"
    return x

df["config_a"] = df["config_a"].apply(clean_config)
df["config_b"] = df["config_b"].apply(clean_config)

keep_cols = [
    "metric",
    "config_a",
    "config_b",
    "mean_a",
    "mean_b",
    "mean_diff",
    "p_raw",
    "p_bonferroni",
    "p_bh",
    "sig_label",
    "cliffs_delta",
    "effect_magnitude",
]

df = df[keep_cols]

round_cols = ["mean_a", "mean_b", "mean_diff", "p_raw", "p_bonferroni", "p_bh", "cliffs_delta"]
for col in round_cols:
    df[col] = df[col].round(4)

out = Path("evaluation/results/retrieval_pairwise_tests_appendix.csv")
out.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(out, index=False)

print(f"Saved appendix-ready table to: {out}")
print(f"Rows: {len(df)}")
