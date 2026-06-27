"""CLI entry point for the full end-to-end pipeline."""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger
from wf_analysis.config import PipelineConfig
from wf_analysis.data import DataLoader, DataCleaner, DataExporter
from wf_analysis.features import (
    DemographicTransformer,
    CategoricalTransformer,
    TemporalTransformer,
    EmbeddingTransformer,
)
from wf_analysis.imputation import EnsembleImputer
from wf_analysis.utils.logging import setup_logging


def main():
    parser = argparse.ArgumentParser(description="Run the Workforce Data Analysis pipeline")
    parser.add_argument("--config", default="config/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    setup_logging(level="INFO", log_file="logs/pipeline.log")
    logger.info(f"Loading config from {args.config}")
    config = PipelineConfig.from_yaml(args.config)

    logger.info("Phase 1: Loading dataset")
    df = DataLoader.load(
        path=config.data.raw_path,
        validate=True,
        schema_path=config.data.schema_path,
    )
    logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")

    logger.info("Phase 2: Cleaning data")
    df = DataCleaner.remove_pii(df, config.data.pii_columns)
    df = DataCleaner.standardize_dates(df, config.data.date_columns)

    logger.info("Phase 3: Feature engineering")
    df = DemographicTransformer().fit(df).transform(df)
    df = CategoricalTransformer().fit(df).transform(df)
    df = TemporalTransformer().fit(df).transform(df)
    df = EmbeddingTransformer(n_components=5).fit(df).transform(df)
    logger.info(f"After features: {df.shape[1]} columns")

    logger.info("Phase 4: Imputation")
    feature_cols = ["JobFamily", "SeniorityLevel", "GenderCode", "Performance Score"]
    imp = EnsembleImputer(
        models=[
            ("statistical", {"method": "median"}),
            ("predictive", {"model_type": "rf"}),
            ("predictive", {"model_type": "gbm"}),
        ],
        weights=[0.2, 0.4, 0.4],
    )
    imp.fit(df, target_column="Age", feature_columns=feature_cols)
    df = imp.impute(df)
    logger.info(f"Age missing after imputation: {df['Age'].isna().sum()}")

    logger.info("Phase 4b: Re-derive DOB-dependent columns after imputation")
    if "Age" in df.columns:
        today = pd.Timestamp.today()
        age_nan = df["BirthYear"].isna() if "BirthYear" in df.columns else True
        if "BirthYear" in df.columns:
            df.loc[age_nan, "BirthYear"] = (today.year - df.loc[age_nan, "Age"]).round(0)
        else:
            df["BirthYear"] = today.year - df["Age"]
        df["BirthYear"] = df["BirthYear"].fillna(0).astype(int)
        by = df["BirthYear"]
        conditions = [
            by <= 1945,
            (by >= 1946) & (by <= 1964),
            (by >= 1965) & (by <= 1980),
            (by >= 1981) & (by <= 1996),
            by >= 1997,
        ]
        choices = ["Silent", "Boomer", "GenX", "Millennial", "GenZ"]
        df["Generation"] = np.select(conditions, choices, default="Unknown")
        bins = [0, 29, 39, 49, 59, 69, 120]
        labels = ["<30", "30s", "40s", "50s", "60s", "70+"]
        df["AgeGroup"] = pd.cut(
            df["Age"], bins=bins, labels=labels, right=False
        )
        df["DOBYear"] = df["DOBYear"].fillna(df["BirthYear"])
        df["DOBMonth"] = df["DOBMonth"].fillna(6)
        df["DOBQuarter"] = df["DOBQuarter"].fillna(2)
    logger.info(f"Missing after refresh: Age={df['Age'].isna().sum()}, BirthYear={df['BirthYear'].isna().sum()}, AgeGroup={df['AgeGroup'].isna().sum()}")

    logger.info("Phase 5: NLP analysis")
    from wf_analysis.nlp import SentimentAnalyzer, TopicModeler, KeywordExtractor

    if "TerminationDescription" in df.columns:
        descriptions = df["TerminationDescription"].dropna()
        descriptions = descriptions[descriptions.str.strip() != ""]
        if len(descriptions) > 0:
            sa = SentimentAnalyzer()
            sentiment = sa.analyze(descriptions)
            for col in sentiment.columns:
                df.loc[sentiment.index, col] = sentiment[col].values

            tm = TopicModeler(n_topics=5)
            tm.fit(descriptions)
            topic_info = tm.get_topic_info()
            logger.info(f"NLP topics found: {len(topic_info)}")

            ke = KeywordExtractor()
            keywords = ke.extract(descriptions.head(200))
            total_phrases = sum(len(v) for v in keywords.values())
            logger.info(f"Keywords extracted: {total_phrases}")

    logger.info("Phase 6: Analysis modules")
    from wf_analysis.analysis import (
        AttritionAnalysis,
        DiversityAnalysis,
        PerformanceAnalysis,
        CompensationAnalysis,
        NetworkAnalysis,
    )

    results = {}
    for name, analyzer in [
        ("attrition", AttritionAnalysis()),
        ("diversity", DiversityAnalysis()),
        ("performance", PerformanceAnalysis()),
        ("compensation", CompensationAnalysis()),
        ("network", NetworkAnalysis()),
    ]:
        try:
            results[name] = analyzer.run(df)
            logger.info(f"{name} analysis complete")
        except Exception as e:
            logger.warning(f"{name} analysis failed: {e}")

    logger.info("Phase 7: Exporting processed data")
    output_dir = Path(config.data.processed_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    DataExporter.to_parquet(df, output_dir / "workforce_clean_base.parquet")
    DataExporter.to_csv(df, output_dir / "workforce_clean_base.csv")
    logger.info(f"Exported {df.shape[0]} rows x {df.shape[1]} cols to {output_dir}")

    print(f"\nPipeline complete!")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Age missing: {df['Age'].isna().sum()}")


if __name__ == "__main__":
    main()
