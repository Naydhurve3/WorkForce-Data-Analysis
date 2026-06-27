"""Test ensemble imputation and KS test results."""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from loguru import logger
logger.remove()

from wf_analysis.config import PipelineConfig
from wf_analysis.data import DataLoader, DataCleaner
from wf_analysis.features import DemographicTransformer, CategoricalTransformer
from wf_analysis.imputation.ensemble import EnsembleImputer

cfg = PipelineConfig()
df = DataLoader.load(cfg.data.raw_path, validate=False)
df = DataCleaner.remove_pii(df, columns=cfg.data.pii_columns)
df = DataCleaner.standardize_dates(df, columns=cfg.data.date_columns)

demo = DemographicTransformer().fit(df)
df = demo.transform(df)
cat = CategoricalTransformer().fit(df)
df = cat.transform(df)

original_parsed = df["Age"].dropna().copy()
print(f"Parsed ages from DOB: {len(original_parsed)}, missing: {df['Age'].isna().sum()}")

imp = EnsembleImputer(
    models=[
        ("statistical", {"method": "median"}),
        ("predictive", {"model_type": "rf"}),
        ("predictive", {"model_type": "gbm"}),
    ],
    weights=[0.2, 0.4, 0.4],
    jitter_range=[-2.0, 2.0],
    use_distribution_match=True,
)
imp.fit(
    df, target_column="Age",
    feature_columns=["JobFamily", "SeniorityLevel", "GenderCode", "Performance Score"],
)
df_imp = imp.impute(df)

remaining = df_imp["Age"].isna().sum()
print(f"Remaining missing: {remaining}")

imputed_ages = df_imp.loc[df["Age"].isna(), "Age"]
print(f"Imputed ages: {len(imputed_ages)}")
ks_stat, ks_pval = ks_2samp(original_parsed, imputed_ages)
print(f"KS statistic: {ks_stat:.4f}")
print(f"KS p-value: {ks_pval:.4f}")
print(f"Same distribution: {ks_pval > 0.05}")
print(f"Original mean={original_parsed.mean():.1f} std={original_parsed.std():.1f}")
print(f"Imputed  mean={imputed_ages.mean():.1f} std={imputed_ages.std():.1f}")
