"""
Bootstrap Ranking Stability: Measure Impact Score ranking stability
via Jaccard overlap and Spearman correlation across 100 bootstrap resamples.
"""
import sys, warnings, json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

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
N_BOOTSTRAP = 100
N_TOP = 30  # track top-30 pairs for stability
RANDOM_SEED = 42

cfg = InteractionConfig()

raw = DataLoader.load(cfg.raw_path, validate=False)
eng = FeatureEngineer(cfg)
fdf = eng.compute_all(raw)
miner = InteractionMiner(cfg)
miner.outcomes = miner.derive_outcomes(raw, fdf)

ir = pd.read_parquet(OUT_DIR / "interaction_results.parquet")
outcomes_list = ir["outcome"].unique()

# Identify top-N pairs per outcome from full data
top_pairs_by_outcome = {}
for oc in outcomes_list:
    sub = ir[ir["outcome"] == oc].head(N_TOP)
    top_pairs_by_outcome[oc] = list(zip(sub["feature_1"], sub["feature_2"]))

logger.info(f"Bootstrap stability: {N_BOOTSTRAP} resamples across {len(outcomes_list)} outcomes")
logger.info(f"  Tracking top-{N_TOP} pairs per outcome")

rng = np.random.RandomState(RANDOM_SEED)
n = len(raw)

# Store rankings for each bootstrap
all_rankings = {}  # outcome -> list of (bootstrap_idx, pair, impact_score)

for b in range(N_BOOTSTRAP):
    if (b + 1) % 10 == 0:
        logger.info(f"  Bootstrap {b+1}/{N_BOOTSTRAP}")

    # Bootstrap resample
    idx = rng.choice(n, size=n, replace=True)
    boot_raw = raw.iloc[idx]
    boot_fdf = fdf.iloc[idx]

    # Reset index to avoid duplicate index issues in crosstab
    boot_raw_r = boot_raw.reset_index(drop=True)
    boot_fdf_r = boot_fdf.reset_index(drop=True)

    for oc in outcomes_list:
        y = miner.outcomes[oc].iloc[idx].reset_index(drop=True)
        pairs = top_pairs_by_outcome[oc]
        scores = []

        for f1, f2 in pairs:
            if f1 not in boot_fdf_r.columns or f2 not in boot_fdf_r.columns:
                continue
            res = miner.test_pair(boot_fdf_r[f1], boot_fdf_r[f2], y)
            if res is not None:
                scores.append({"pair": f"{f1} x {f2}", "impact": res["impact"]})

        if scores:
            sdf = pd.DataFrame(scores).sort_values("impact", ascending=False)
            sdf["rank"] = range(1, len(sdf) + 1)
            key = f"{b}_{oc}"
            for _, r in sdf.iterrows():
                all_rankings.setdefault(oc, []).append({
                    "bootstrap": b, "pair": r["pair"], "impact": r["impact"], "rank": r["rank"]
                })

# Compute stability metrics per outcome
results_rows = []
for oc in outcomes_list:
    if oc not in all_rankings:
        continue
    rankings = pd.DataFrame(all_rankings[oc])
    pairs = rankings["pair"].unique()

    # Build rank matrix: pairs × bootstraps
    rank_matrix = rankings.pivot_table(index="pair", columns="bootstrap", values="rank")

    # Jaccard overlap: for each pair of bootstraps, what fraction of top-K overlap?
    jaccard_scores_10 = []
    jaccard_scores_20 = []
    spearman_scores = []

    boot_list = sorted(rank_matrix.columns)
    for i, b1 in enumerate(boot_list):
        top10_b1 = set(rank_matrix[rank_matrix[b1] <= 10].index)
        top20_b1 = set(rank_matrix[rank_matrix[b1] <= 20].index)
        ranks_b1 = rank_matrix[b1]
        for b2 in boot_list[i + 1:]:
            top10_b2 = set(rank_matrix[rank_matrix[b2] <= 10].index)
            top20_b2 = set(rank_matrix[rank_matrix[b2] <= 20].index)

            # Jaccard
            inter10 = len(top10_b1 & top10_b2)
            union10 = len(top10_b1 | top10_b2)
            if union10 > 0:
                jaccard_scores_10.append(inter10 / union10)

            inter20 = len(top20_b1 & top20_b2)
            union20 = len(top20_b1 | top20_b2)
            if union20 > 0:
                jaccard_scores_20.append(inter20 / union20)

            # Spearman on common pairs
            common = ranks_b1.index.intersection(rank_matrix[b2].index)
            if len(common) >= 5:
                r_s, _ = spearmanr(ranks_b1[common], rank_matrix[b2].loc[common])
                if not np.isnan(r_s):
                    spearman_scores.append(r_s)

    j10 = round(float(np.mean(jaccard_scores_10)), 4) if jaccard_scores_10 else 0
    j10_std = round(float(np.std(jaccard_scores_10)), 4) if jaccard_scores_10 else 0
    j20 = round(float(np.mean(jaccard_scores_20)), 4) if jaccard_scores_20 else 0
    j20_std = round(float(np.std(jaccard_scores_20)), 4) if jaccard_scores_20 else 0
    sp = round(float(np.mean(spearman_scores)), 4) if spearman_scores else 0
    sp_std = round(float(np.std(spearman_scores)), 4) if spearman_scores else 0

    results_rows.append({
        "outcome": oc,
        "n_bootstrap": N_BOOTSTRAP,
        "n_pairs_tracked": len(pairs),
        "jaccard_top10_mean": j10,
        "jaccard_top10_std": j10_std,
        "jaccard_top20_mean": j20,
        "jaccard_top20_std": j20_std,
        "spearman_mean": sp,
        "spearman_std": sp_std,
    })
    logger.info(f"  {oc:20s}: Jaccard@10={j10:.3f}+-{j10_std:.3f} Jaccard@20={j20:.3f}+-{j20_std:.3f} Spearman={sp:.3f}+-{sp_std:.3f}")

rdf = pd.DataFrame(results_rows)
rdf.to_csv(OUT_DIR / "bootstrap_stability.csv", index=False)
logger.info(f"\nBootstrap stability saved to {OUT_DIR / 'bootstrap_stability.csv'}")
