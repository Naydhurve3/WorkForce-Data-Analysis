"""Phase 3: Interaction Mining — 351 pairs × 5 outcomes = 1,755 significance tests."""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.mining import InteractionMiner
from wf_analysis.interaction.figures import EDAFigureFactory


def main():
    logger.info("=" * 60)
    logger.info("  Phase 3: Interaction Mining — 1,755 Statistical Tests")
    logger.info("=" * 60)

    cfg = InteractionConfig()

    raw_df = DataLoader.load(cfg.raw_path, validate=False)
    logger.info(f"Loaded: {raw_df.shape[0]} rows x {raw_df.shape[1]} columns")

    engineer = FeatureEngineer(cfg)
    feature_df = engineer.compute_all(raw_df)
    logger.info(f"Feature matrix: {feature_df.shape}")

    miner = InteractionMiner(cfg)
    results = miner.run(raw_df, feature_df)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(results, pd.DataFrame) and len(results) > 0:
        results.to_parquet(out_dir / "interaction_results.parquet", index=False)
        results.head(50).to_json(out_dir / "interaction_top50.json", orient="records", indent=2)
        summary = {
            "total_tests": len(results),
            "significant_p05": int(results["significant"].sum()),
            "significant_bonferroni": int(results["bonferroni"].sum()),
            "outcomes": list(results["outcome"].unique()),
            "methods_used": list(results["method"].value_counts().to_dict()),
            "top_pair": results.iloc[0].to_dict() if len(results) > 0 else None,
        }
        with open(out_dir / "interaction_summary.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        logger.info(f"Saved {len(results)} interaction results")
        logger.info(f"  Significant (p<0.05): {summary['significant_p05']}")
        logger.info(f"  Bonferroni: {summary['significant_bonferroni']}")
    else:
        logger.error("Interaction mining produced no results")
        return

    top_result = results.iloc[0].to_dict()
    tree_data = miner.run_segmentation(feature_df, top_result["outcome"],
                                        top_result["feature_1"], top_result["feature_2"])

    figures = EDAFigureFactory(cfg)
    paths = []

    p1 = figures.figure_16_interaction_heatmap(results)
    paths.append(p1)
    logger.info(f"Figure 16: Interaction heatmap -> {p1}")

    p2 = figures.figure_17_top_interactions(results)
    paths.append(p2)
    logger.info(f"Figure 17: Top interactions -> {p2}")

    p3 = figures.figure_18_interaction_pairplot(feature_df, top_result)
    paths.append(p3)
    logger.info(f"Figure 18: Interaction pair plot -> {p3}")

    if tree_data is not None:
        p4 = figures.figure_19_tree_segmentation(tree_data, feature_df, top_result)
        paths.append(p4)
        logger.info(f"Figure 19: Tree segmentation -> {p4}")
    else:
        logger.warning("Skipping tree segmentation (insufficient data)")

    p5 = figures.figure_20_mutual_information_matrix(results)
    paths.append(p5)
    logger.info(f"Figure 20: MI matrix -> {p5}")

    p6 = figures.figure_21_interaction_network(results)
    paths.append(p6)
    logger.info(f"Figure 21: Interaction network -> {p6}")

    p7 = figures.figure_22_outcome_ranking(results)
    paths.append(p7)
    logger.info(f"Figure 22: Outcome ranking -> {p7}")

    p8 = figures.figure_23_interaction_dashboard(results)
    paths.append(p8)
    logger.info(f"Figure 23: Interaction dashboard -> {p8}")

    logger.info("=" * 60)
    logger.info(f"  Phase 3 Complete: {len(paths)} figures generated")
    logger.info(f"  Total tests: {len(results)}, Significant: {summary['significant_p05']}")
    logger.info("=" * 60)
    return paths


if __name__ == "__main__":
    main()
