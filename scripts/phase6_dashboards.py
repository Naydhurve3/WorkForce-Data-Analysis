"""Phase 6: Dashboards — 8 composite figures (53–60) synthesizing all prior phases."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import pandas as pd
from loguru import logger

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.figures import EDAFigureFactory


def main():
    logger.info("=" * 60)
    logger.info("  Phase 6: Dashboards — 8 composite figures (53–60)")
    logger.info("=" * 60)

    cfg = InteractionConfig()

    raw_df = DataLoader.load(cfg.raw_path, validate=False)
    engineer = FeatureEngineer(cfg)
    feature_df = engineer.compute_all(raw_df)
    df = pd.concat([raw_df, feature_df.drop(columns=[c for c in raw_df.columns if c in feature_df.columns])], axis=1)

    ff = EDAFigureFactory(cfg)

    interaction_results = None
    ir_path = Path(cfg.output_dir) / "interaction_results.parquet"
    if ir_path.exists():
        interaction_results = pd.read_parquet(ir_path)
        logger.info(f"Loaded interaction results ({len(interaction_results)} pairs)")

    logger.info("Figure 53: Attrition Dashboard")
    ff.figure_53_attrition_dashboard(df)

    logger.info("Figure 54: Compensation Dashboard")
    ff.figure_54_compensation_dashboard(df)

    logger.info("Figure 55: Diversity Dashboard")
    ff.figure_55_diversity_dashboard(df)

    logger.info("Figure 56: Performance Dashboard")
    ff.figure_56_performance_dashboard(df)

    logger.info("Figure 57: Career Progression Dashboard")
    ff.figure_57_career_dashboard(df)

    logger.info("Figure 58: Interaction Matrix")
    ff.figure_58_interaction_matrix(interaction_results)

    model_results = {}
    mr_path = Path(cfg.output_dir) / "model_results.json"
    if mr_path.exists():
        with open(mr_path) as f:
            model_results = json.load(f)
        logger.info(f"Loaded model results ({len(model_results)} outcomes)")

    logger.info("Figure 59: ROC Curves")
    ff.figure_59_roc_curves(model_results)

    logger.info("Figure 60: Executive Summary")
    ff.figure_60_executive_summary(df)

    logger.info("=" * 60)
    logger.info("  Phase 6 complete — 8 figures (53–60)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
