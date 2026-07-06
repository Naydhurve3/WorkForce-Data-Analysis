"""
Binning Sensitivity: Vary bin count for continuous feature discretization
and measure Impact Score ranking stability.
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.mining import InteractionMiner

warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stdout, format="{message}")

OUT_DIR = Path("data") / "interaction"
cfg = InteractionConfig()

raw = DataLoader.load(cfg.raw_path, validate=False)
eng = FeatureEngineer(cfg)
fdf = eng.compute_all(raw)
miner = InteractionMiner(cfg)
miner.outcomes = miner.derive_outcomes(raw, fdf)

BIN_COUNTS = [5, 10, 15, 20]
OUTCOMES = list(miner.outcomes.columns)
OUTCOME_FOCUS = "is_terminated"  # primary outcome

logger.info(f"Binning sensitivity testing: {BIN_COUNTS}")
logger.info(f"  Focus outcome: {OUTCOME_FOCUS}")

all_results = {}

for n_bins in BIN_COUNTS:
    logger.info(f"  Binning with n_bins={n_bins}...")
    # Discretize numeric features
    fdf_binned = fdf.copy()
    for c in fdf_binned.columns:
        if pd.api.types.is_numeric_dtype(fdf_binned[c]) and fdf_binned[c].nunique() > n_bins:
            fdf_binned[c] = pd.cut(fdf_binned[c], bins=n_bins, labels=False)

    results = miner.test_all_pairs(fdf_binned, miner.outcomes)
    results = results[results["outcome"] == OUTCOME_FOCUS]
    results = results.sort_values("impact", ascending=False).reset_index(drop=True)
    results["rank"] = range(1, len(results) + 1)
    all_results[n_bins] = results
    logger.info(f"    n_bins={n_bins}: {len(results)} pairs, top={results.iloc[0]['feature_1']} x {results.iloc[0]['feature_2']} (impact={results.iloc[0]['impact']:.2f})")

# Load reference (non-binned) results
ref = pd.read_parquet(OUT_DIR / "interaction_results.parquet")
ref = ref[ref["outcome"] == OUTCOME_FOCUS].sort_values("impact", ascending=False).reset_index(drop=True)
ref["rank"] = range(1, len(ref) + 1)

# Compute rank correlations
ref_pairs = ref[["feature_1", "feature_2", "impact", "rank"]].copy()
ref_pairs["pair"] = ref_pairs["feature_1"] + " x " + ref_pairs["feature_2"]
ref_dict = ref_pairs.set_index("pair")["rank"].to_dict()

rows = []
for n_bins, bin_df in all_results.items():
    bin_df["pair"] = bin_df["feature_1"] + " x " + bin_df["feature_2"]
    bin_dict = bin_df.set_index("pair")["rank"].to_dict()

    # Spearman on common pairs
    common = set(ref_dict.keys()) & set(bin_dict.keys())
    if len(common) >= 5:
        ref_ranks = [ref_dict[p] for p in common]
        bin_ranks = [bin_dict[p] for p in common]
        sp, _ = spearmanr(ref_ranks, bin_ranks)
        sp = round(float(sp), 4)
    else:
        sp = 0

    # Jaccard overlap on top-10 and top-20
    ref_top10 = set(ref_df.head(10)["pair"] if (ref_df := ref_pairs) is not None else [])  # noqa
    # Actually compute properly
    ref_top10 = set(ref_pairs.head(10)["pair"])
    ref_top20 = set(ref_pairs.head(20)["pair"])
    bin_top10 = set(bin_df.head(10)["pair"])
    bin_top20 = set(bin_df.head(20)["pair"])

    j10 = round(len(ref_top10 & bin_top10) / max(len(ref_top10 | bin_top10), 1), 4)
    j20 = round(len(ref_top20 & bin_top20) / max(len(ref_top20 | bin_top20), 1), 4)

    rows.append({
        "n_bins": n_bins,
        "n_pairs_tested": len(bin_df),
        "spearman_vs_reference": sp,
        "jaccard_top10": j10,
        "jaccard_top20": j20,
        "top_pair": f"{bin_df.iloc[0]['feature_1']} x {bin_df.iloc[0]['feature_2']}",
        "top_impact": round(float(bin_df.iloc[0]["impact"]), 4),
    })
    logger.info(f"  n_bins={n_bins}: Spearman={sp:.4f} Jaccard@10={j10:.4f} Jaccard@20={j20:.4f}")

rdf = pd.DataFrame(rows)
rdf.to_csv(OUT_DIR / "binning_sensitivity.csv", index=False)
logger.info(f"\nBinning sensitivity saved to {OUT_DIR / 'binning_sensitivity.csv'}")
