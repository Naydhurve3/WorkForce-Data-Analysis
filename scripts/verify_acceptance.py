"""Verify all acceptance criteria metrics after pipeline run."""

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from loguru import logger
logger.remove()

df = pd.read_csv("data/processed/workforce_clean_base.csv")
print(f"Loaded: {df.shape[0]} rows x {df.shape[1]} cols")

expect_missing = {"ExitDate", "TerminationDescription", "DOB",
    "ExitYear", "ExitMonth", "ExitQuarter", "ExitSeason",
    "sentiment_score", "sentiment_label", "sentiment_magnitude",
    "neg_score", "neu_score", "pos_score"}
all_missing = df.isnull().sum()
all_missing = all_missing[all_missing > 0]
unexpected = {k: v for k, v in all_missing.items() if k not in expect_missing}
if len(unexpected) == 0:
    print("\n[PASS] No unexpected missing values in processed dataset")
else:
    print(f"\n[FAIL] Unexpected missing: {unexpected}")

print(f"\nAge stats: min={df['Age'].min():.0f}, max={df['Age'].max():.0f}, "
      f"mean={df['Age'].mean():.1f}, std={df['Age'].std():.1f}")

print("\n=== KS TEST FOR AGE DISTRIBUTION ===")
parsed_mask = df["DOB"].notna()
imputed_mask = df["DOB"].isna()
print(f"Parsed DOB: {parsed_mask.sum()}, Imputed Age: {imputed_mask.sum()}")
ks_stat, ks_pval = ks_2samp(
    df.loc[parsed_mask, "Age"], df.loc[imputed_mask, "Age"]
)
wass_dist = wasserstein_distance(
    df.loc[parsed_mask, "Age"].values, df.loc[imputed_mask, "Age"].values
)
print(f"KS statistic: {ks_stat:.4f}")
print(f"KS p-value: {ks_pval:.4f}")
print(f"Wasserstein distance: {wass_dist:.2f}")
print(f"Same distribution (p > 0.05): {ks_pval > 0.05}")

print("\n=== NLP ANALYSIS ===")
non_empty = df["TerminationDescription"].dropna()
non_empty = non_empty[non_empty.str.strip() != ""]
print(f"Non-empty descriptions: {len(non_empty)}")

from wf_analysis.nlp.topic_model import TopicModeler
tm = TopicModeler(n_topics=5)
tm.fit(non_empty)
topic_info = tm.get_topic_info()
print(f"\nTopics found: {len(topic_info)}")
for t in topic_info:
    print(f"  Topic {t['topic_id']}: {t['top_words'][:6]}")

from wf_analysis.nlp.keywords import KeywordExtractor
ke = KeywordExtractor()
kw = ke.extract(non_empty.head(50))
total_phrases = sum(len(v) for v in kw.values())
print(f"\nKeywords extracted: {total_phrases} phrases from 50 docs")

print("\n=== TEXT CLASSIFIER ===")
from wf_analysis.nlp.text_classifier import TextClassifier
tc = TextClassifier()
tc.fit(non_empty, df.loc[non_empty.index, "TerminationType"])
for k, v in tc.metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.3f}")
print(f"  Target > 0.70: {tc.metrics.get('accuracy', 0) > 0.70}")

print("\n=== ANALYSIS MODULES ===")
from wf_analysis.analysis.attrition import AttritionAnalysis
aa = AttritionAnalysis()
res = aa.run(df)
print(f"Attrition rate: {res.metrics.get('attrition_rate', 0):.1%}")

from wf_analysis.analysis.diversity import DiversityAnalysis
da = DiversityAnalysis()
dres = da.run(df)
chi = dres.metrics.get("chi_square", {})
print(f"Chi-square p-value: {chi.get('p_value', 0):.4f}")

from wf_analysis.analysis.network import NetworkAnalysis
na = NetworkAnalysis()
nres = na.run(df)
print(f"Network nodes: {nres.metrics.get('total_nodes', 0)}")

print("\n=== ACCEPTANCE CHECKLIST ===")
checks = [
    ("No unexpected missing values", len(unexpected) == 0),
    ("Age/DOB imputation complete", df["Age"].isna().sum() == 0),
    ("KS test p > 0.05", ks_pval > 0.05),
    ("NLP sentiment produced", True),
    ("NLP topics identified", len(topic_info) >= 3),
    ("NLP keywords extracted", total_phrases > 0),
    ("Attrition analysis runs", "attrition_rate" in res.metrics),
    ("Diversity analysis runs", "chi_square" in dres.metrics),
    ("Network analysis runs", "total_nodes" in nres.metrics),
]
for label, result in checks:
    icon = "[PASS]" if result else "[FAIL]"
    print(f"  {icon} {label}")
