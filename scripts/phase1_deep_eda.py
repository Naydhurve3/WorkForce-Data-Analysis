"""Phase 1: Deep EDA — univariate stats, missing patterns, outliers, PCA, t-SNE, correlation networks."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.deep_eda import DeepEDA
from wf_analysis.interaction.dim_reduction import DimReduction
from wf_analysis.interaction.figures import EDAFigureFactory


def parse_dates(df, date_cols):
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], format="mixed", errors="coerce")
    return df


def main():
    logger.info("=" * 60)
    logger.info("  Phase 1: Deep EDA — Feature Interaction Framework")
    logger.info("=" * 60)

    cfg = InteractionConfig()

    df = DataLoader.load(cfg.raw_path, validate=False)
    logger.info(f"Loaded: {df.shape[0]} rows x {df.shape[1]} columns")
    df = parse_dates(df, cfg.date_columns)
    df = df.drop(columns=[cfg.id_column], errors="ignore")

    eda = DeepEDA(cfg)
    eda_results = eda.run(df)

    dimred = DimReduction(cfg)
    dimred_results = dimred.run_all(df)

    figures = EDAFigureFactory(cfg)

    paths = []

    p1 = figures.figure_1_univariate_grid(df, eda_results["stats"])
    paths.append(p1)
    logger.info(f"Figure 1: Univariate distribution grid -> {p1}")

    p2 = figures.figure_2_categorical_grid(df, eda_results["stats"])
    paths.append(p2)
    logger.info(f"Figure 2: Categorical distribution grid -> {p2}")

    p3 = figures.figure_3_missing_patterns(eda_results["missing_matrix"])
    paths.append(p3)
    logger.info(f"Figure 3: Missing pattern heatmap -> {p3}")

    p4 = figures.figure_4_missing_correlation(eda_results["missing_corr"])
    paths.append(p4)
    logger.info(f"Figure 4: Missing correlation -> {p4}")

    p5 = figures.figure_5_outlier_comparison(eda_results["outliers"], df)
    paths.append(p5)
    logger.info(f"Figure 5: Outlier method comparison -> {p5}")

    p6 = figures.figure_6_outlier_consensus(eda_results["outliers"], df)
    paths.append(p6)
    logger.info(f"Figure 6: Outlier consensus -> {p6}")

    p7 = figures.figure_7_correlation_network(dimred_results["network"])
    paths.append(p7)
    logger.info(f"Figure 7: Correlation network -> {p7}")

    p8 = figures.figure_8_pca_scree(dimred_results["pca"])
    paths.append(p8)
    logger.info(f"Figure 8: PCA scree -> {p8}")

    p9 = figures.figure_9_pca_biplot(dimred_results["pca"], df)
    paths.append(p9)
    logger.info(f"Figure 9: PCA biplot -> {p9}")

    p10 = figures.figure_10_tsne_landscape(dimred_results["tsne"], df)
    paths.append(p10)
    logger.info(f"Figure 10: t-SNE landscape -> {p10}")

    p11 = figures.figure_11_silhouette_comparison(dimred_results["silhouette"])
    paths.append(p11)
    logger.info(f"Figure 11: Silhouette comparison -> {p11}")

    p12 = figures.figure_12_dashboard_summary(eda_results["stats"], eda_results["outliers"])
    paths.append(p12)
    logger.info(f"Figure 12: EDA dashboard summary -> {p12}")

    logger.info("=" * 60)
    logger.info(f"  Phase 1 Complete: {len(paths)} figures generated")
    logger.info(f"  Output: {cfg.figure_dir}/")
    logger.info(f"  Report: {cfg.output_dir}/eda_summary.json")
    logger.info("=" * 60)

    return paths


if __name__ == "__main__":
    main()
