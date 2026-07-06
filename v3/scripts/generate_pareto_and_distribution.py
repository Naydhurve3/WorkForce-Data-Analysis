import pandas as pd
import matplotlib.pyplot as plt
import json
import numpy as np
import os

out_dir = "research_paper/02_figures_curated"
os.makedirs(out_dir, exist_ok=True)

# --- Figure 1: Pareto Lorenz Curve ---
pareto = pd.read_csv("../data/interaction/pareto_analysis.csv")
pareto = pareto.sort_values("impact", ascending=False).reset_index(drop=True)
pareto["cumulative_pct"] = pareto["impact"].cumsum() / pareto["impact"].sum() * 100
pareto["pair_pct"] = (np.arange(len(pareto)) + 1) / len(pareto) * 100

fig, ax = plt.subplots(figsize=(5, 4))
ax.plot(pareto["pair_pct"], pareto["cumulative_pct"], "b-", linewidth=2)
ax.plot([0, 100], [0, 100], "k--", alpha=0.3, label="Uniform (diagonal)")

# Annotate key thresholds
for x, y, label in [(1.3, 50, "23 pairs → 50%"), (3.1, 80, "54 pairs → 80%")]:
    ax.plot(x, y, "ro", markersize=5)
    ax.annotate(label, (x, y), xytext=(x + 5, y - 8), fontsize=8, arrowprops=dict(arrowstyle="->", color="red"))

ax.set_xlabel("Pairs tested (%)", fontsize=10)
ax.set_ylabel("Cumulative impact (%)", fontsize=10)
ax.set_title("Pareto Lorenz Curve: Interaction Impact Concentration", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/pareto_lorenz.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved pareto_lorenz.png")

# --- Figure 2: Impact Score Distribution (log-log) ---
props = pd.read_csv("../data/interaction/impact_score_properties.csv")
all_impacts = []
for col in props.columns:
    if col.endswith("_impact_mean") or col == "impact_mean":
        vals = props[col].values
        all_impacts.extend(vals[vals > 0])

all_impacts = np.array(all_impacts)
all_impacts = all_impacts[all_impacts > 1e-10]

fig, ax = plt.subplots(figsize=(5, 4))
bins = np.logspace(np.log10(all_impacts.min()), np.log10(all_impacts.max()), 50)
ax.hist(all_impacts, bins=bins, alpha=0.7, color="steelblue", edgecolor="white")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Impact Score (log scale)", fontsize=10)
ax.set_ylabel("Frequency (log scale)", fontsize=10)
ax.set_title("Interaction Impact Score Distribution", fontsize=10)
ax.axvline(x=np.median(all_impacts), color="red", linestyle="--", linewidth=1, label=f"Median = {np.median(all_impacts):.4f}")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out_dir}/impact_distribution.png", dpi=200, bbox_inches="tight")
plt.close()
print("Saved impact_distribution.png")
