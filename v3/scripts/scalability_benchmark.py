"""
Scalability Benchmark: Measure Impact Score runtime as feature count increases.
Tests at 10, 20, 40, 60, 80 features to show O(p^2) scaling.
"""
import sys, time, warnings
from pathlib import Path
import numpy as np
import pandas as pd

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

all_features = fdf.columns.tolist()
N_FEATURES = [5, 10, 15, 20, 27]

logger.info("Scalability Benchmark: Impact Score Runtime vs Feature Count")
logger.info("=" * 60)

results = []
for n_feat in N_FEATURES:
    feat_subset = all_features[:n_feat]
    fdf_sub = fdf[feat_subset]
    outcome_sub = miner.outcomes[["is_terminated"]]

    n_pairs = n_feat * (n_feat - 1) // 2
    n_tests = n_pairs

    t0 = time.time()
    miner.test_all_pairs(fdf_sub, outcome_sub)
    elapsed = round(time.time() - t0, 2)

    logger.info(f"  Features={n_feat:2d}  Pairs={n_pairs:4d}  Tests={n_tests:5d}  Runtime={elapsed:7.2f}s  Per-pair={elapsed/n_pairs:.4f}s")
    results.append({
        "n_features": n_feat,
        "n_pairs": n_pairs,
        "n_tests": n_tests,
        "runtime_seconds": elapsed,
        "seconds_per_pair": round(elapsed / n_pairs, 4),
    })

res_df = pd.DataFrame(results)
res_df.to_csv(OUT_DIR / "scalability_benchmark.csv", index=False)
logger.info(f"\nSaved to {OUT_DIR / 'scalability_benchmark.csv'}")
