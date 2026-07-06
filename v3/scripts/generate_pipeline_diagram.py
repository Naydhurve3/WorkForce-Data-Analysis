"""Generate a pipeline architecture diagram for the research paper."""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "research_paper" / "02_figures_curated"

fig, ax = plt.subplots(figsize=(16, 6))
ax.set_xlim(0, 16)
ax.set_ylim(0, 6)
ax.axis("off")

phases = [
    ("1\nDeep EDA", "12 figures", "#2E86AB"),
    ("2\nFeature Eng.", "27 features", "#A23B72"),
    ("3\nInteraction\nMining", "1,755 tests", "#F18F01"),
    ("4\nPredictive\nModels", "13 models", "#C73E1D"),
    ("5\nDeep Dives", "16 figures", "#6B4E71"),
    ("6\nDashboards", "8 figures", "#3B8C5E"),
    ("7\nHTML Report", "Jinja2", "#1A535C"),
]

box_w, box_h = 1.8, 1.2
gap = 0.35
start_x = 0.5

for i, (label, sublabel, color) in enumerate(phases):
    x = start_x + i * (box_w + gap)
    y = 3.0
    rect = mpatches.FancyBboxPatch(
        (x, y - box_h / 2), box_w, box_h,
        boxstyle="round,pad=0.1",
        facecolor=color, edgecolor="white", linewidth=2, alpha=0.9
    )
    ax.add_patch(rect)
    ax.text(x + box_w / 2, y + 0.15, label, ha="center", va="center",
            fontsize=9, fontweight="bold", color="white", linespacing=1.3)
    ax.text(x + box_w / 2, y - 0.35, sublabel, ha="center", va="center",
            fontsize=7, color="white", alpha=0.9)

    if i < len(phases) - 1:
        ax.annotate("", xy=(x + box_w + gap * 0.3, y),
                    xytext=(x + box_w - gap * 0.3, y),
                    arrowprops=dict(arrowstyle="->", color="#555", lw=2))

# Title
ax.text(8, 5.5, "WorkForce Data Analysis v2.0 — 7-Phase Pipeline",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#333")

# Stats bar
stats = ["3,000 employees", "53 features (26 raw + 27 engineered)",
         "5 outcomes", "3 model families (LR / RF / XGB)",
         "SHAP + Survival + Deep Dives", "60 publication figures"]
stat_y = 0.8
ax.text(0.2, stat_y, "Key Stats:", fontsize=9, fontweight="bold", color="#333",
        va="center")
for j, s in enumerate(stats):
    ax.text(2.0 + j * 2.3, stat_y, f"  {s}",
            fontsize=7, color="#555", va="center",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#f0f0f0", edgecolor="#ddd"))

plt.tight_layout()
plt.savefig(OUT / "pipeline_overview.png", dpi=200, bbox_inches="tight",
            facecolor="white")
print(f"[OK] pipeline_overview.png saved to {OUT}")
