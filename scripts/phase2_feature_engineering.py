"""Phase 2: Feature Engineering — derive 27 analytical features from raw data."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.figures import EDAFigureFactory


def main():
    logger.info("=" * 60)
    logger.info("  Phase 2: Feature Engineering — 27 Analytical Features")
    logger.info("=" * 60)

    cfg = InteractionConfig()

    df = DataLoader.load(cfg.raw_path, validate=False)
    logger.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")

    engineer = FeatureEngineer(cfg)
    feature_df = engineer.compute_all(df)
    logger.info(f"Feature matrix: {feature_df.shape[0]} rows x {feature_df.shape[1]} columns")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(out_dir / "feature_matrix.parquet", index=False)
    logger.info(f"Saved feature matrix to {out_dir / 'feature_matrix.parquet'}")

    meta = []
    for c in feature_df.columns:
        s = feature_df[c]
        meta.append({
            "feature": c,
            "dtype": str(s.dtype),
            "count": int(s.notna().sum()),
            "missing": int(s.isna().sum()),
            "missing_pct": round(float(s.isna().mean() * 100), 2),
            "unique": int(s.nunique()),
            "type": "numeric" if pd.api.types.is_numeric_dtype(s) else "categorical",
        })

    with open(out_dir / "feature_summary.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    logger.info(f"Saved feature summary to {out_dir / 'feature_summary.json'}")

    figures = EDAFigureFactory(cfg)
    paths = []

    p1 = figures.figure_13_feature_correlation(feature_df)
    paths.append(p1)
    logger.info(f"Figure 13: Feature correlation -> {p1}")

    p2 = figures.figure_14_feature_summary(feature_df)
    paths.append(p2)
    logger.info(f"Figure 14: Feature summary -> {p2}")

    p3 = figures.figure_15_feature_distributions(feature_df)
    paths.append(p3)
    logger.info(f"Figure 15: Feature distributions -> {p3}")

    n_num = sum(1 for m in meta if m["type"] == "numeric")
    n_cat = len(meta) - n_num
    logger.info("=" * 60)
    logger.info(f"  Phase 2 Complete: {len(paths)} figures generated")
    logger.info(f"  Features: {len(meta)} ({n_num} numeric, {n_cat} categorical)")
    logger.info("=" * 60)
    return paths


if __name__ == "__main__":
    main()
