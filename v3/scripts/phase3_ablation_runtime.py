"""
Phase 3: Component Ablation Table & Runtime Comparison.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.mining import InteractionMiner

logger.remove()
logger.add(sys.stdout, format="{message}")

OUT_DIR = Path("data") / "interaction"
cfg = InteractionConfig()

# ── Component Ablation Table ─────────────────────────────────────────────
with open(OUT_DIR / "ablation_ranking.json") as f:
    ablation = json.load(f)

logger.info("=" * 60)
logger.info("COMPONENT ABLATION: Spearman Correlation vs Full Impact Score")
logger.info("=" * 60)

variants_map = {
    "MI only": "MI",
    "Statistic only": "Effect Size",
    "MI x log(n)": "MI x log(n)",
    "Additive (norm)": "Additive (z-score)",
    "Ensemble (norm)": "Ensemble (rank)",
}

agg_rows = []
for oc, oc_data in ablation["per_outcome"].items():
    corrs = oc_data["correlations"]
    for pair, r in sorted(corrs.items()):
        if "Impact (full) vs" in pair:
            variant = pair.replace("Impact (full) vs ", "").strip()
            agg_rows.append({"outcome": oc, "variant": variant, "spearman": r})

agg_df = pd.DataFrame(agg_rows)

# Pivot table: outcomes as columns, variants as rows
pivot = agg_df.pivot_table(index="variant", columns="outcome", values="spearman")
pivot["Mean"] = pivot.mean(axis=1).round(4)
pivot["Min"] = pivot.min(axis=1).round(4)
pivot = pivot.sort_values("Mean", ascending=False)

print(pivot.to_string())
pivot.to_csv(OUT_DIR / "ablation_summary_table.csv")

# Identify best variant
best = pivot["Mean"].idxmax()
logger.info(f"\nBest alternative: {best} (mean Spearman = {pivot.loc[best, 'Mean']:.4f})")
logger.info(f"Full Impact Score is the reference (Spearman=1.0 by definition)")

# ── Runtime Benchmark ────────────────────────────────────────────────────
logger.info("\n" + "=" * 60)
logger.info("RUNTIME BENCHMARK: Impact Score Scaling")
logger.info("=" * 60)

raw = DataLoader.load(cfg.raw_path, validate=False)
eng = FeatureEngineer(cfg)
fdf = eng.compute_all(raw)
miner = InteractionMiner(cfg)
miner.outcomes = miner.derive_outcomes(raw, fdf)

n_sizes = [500, 1000, 3000]
time_rows = []
for n in n_sizes:
    sub_raw = raw.head(n)
    sub_fdf = fdf.loc[sub_raw.index]
    miner_sub = InteractionMiner(cfg)
    miner_sub.outcomes = miner_sub.derive_outcomes(sub_raw, sub_fdf)

    t0 = time.time()
    _ = miner_sub.test_all_pairs(sub_fdf, miner_sub.outcomes)
    elapsed = round(time.time() - t0, 2)

    n_pairs = 351
    per_pair = round(elapsed / n_pairs, 4)
    logger.info(f"  N={n:5d}: {elapsed:6.1f}s total, {per_pair:.4f}s per pair")
    time_rows.append({
        "n_rows": n,
        "n_pairs_tested": n_pairs,
        "time_seconds": elapsed,
        "seconds_per_pair": per_pair,
    })

time_df = pd.DataFrame(time_rows)
time_df.to_csv(OUT_DIR / "runtime_benchmark.csv", index=False)

# ── Fair Comparison Table (theoretical) ──────────────────────────────────
comparison = pd.DataFrame([
    {"Method": "Impact Score", "Model Training Required": "No", "Domain Agnostic": "Yes",
     "Interpretable": "Yes", "Pairwise Test": "Yes", "Runtime (N=3000)": f"{time_rows[-1]['time_seconds']}s" if time_rows else "~30s"},
    {"Method": "SHAP", "Model Training Required": "Yes", "Domain Agnostic": "No (model-specific)",
     "Interpretable": "Partial", "Pairwise Test": "No", "Runtime (N=3000)": "~hours (model + SHAP values)"},
    {"Method": "H-statistic", "Model Training Required": "Yes", "Domain Agnostic": "Partial",
     "Interpretable": "Partial", "Pairwise Test": "Yes", "Runtime (N=3000)": "~0.3s per pair (model not included)"},
])
comparison.to_csv(OUT_DIR / "runtime_comparison_table.csv", index=False)
print(f"\n{comparison.to_string()}")

logger.info("\nPhase 3 complete. Files saved:")
logger.info(f"  {OUT_DIR}/ablation_summary_table.csv")
logger.info(f"  {OUT_DIR}/runtime_benchmark.csv")
logger.info(f"  {OUT_DIR}/runtime_comparison_table.csv")
