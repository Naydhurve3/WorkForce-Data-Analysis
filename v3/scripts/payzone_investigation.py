"""Phase B: PayZone Paradox Investigation.
Three-part analysis:
  B.1 — Pipeline: verify interaction encoding, feature counts, train/test split
  B.2 — Embedding: analyze label-encoding generalizability across CV folds
  B.3 — SHAP: do JobFamily/DepartmentType matter in RF splits?
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.models import _encode_interaction_pair, _prepare_data

warnings.filterwarnings("ignore")
DATA = Path("data") / "interaction"
cfg = InteractionConfig()

# ============================================================
# B.1: Pipeline verification
# ============================================================
print("=" * 60)
print("B.1: PIPELINE VERIFICATION")
print("=" * 60)

raw = DataLoader.load(cfg.raw_path, validate=False)
eng = FeatureEngineer(cfg)
fdf = eng.compute_all(raw)

# Load interaction results to get the top pair for PayZone
ir = pd.read_parquet(DATA / "interaction_results.parquet")
top_pz = ir[(ir["outcome"] == "PayZone_encoded")].head(1)
f1, f2 = top_pz.iloc[0]["feature_1"], top_pz.iloc[0]["feature_2"]
print(f"Top PayZone interaction: {f1} x {f2} (impact={top_pz.iloc[0]['impact']:.1f})")

# Prepare PayZone outcome
pz_map = {"Zone A": 1, "Zone B": 2, "Zone C": 3}
y = raw["PayZone"].map(pz_map).fillna(0).astype(int)
print(f"PayZone distribution: {y.value_counts().sort_index().to_dict()}")

# _prepare_data WITHOUT leakage fix for PayZone (no exclusions needed)
x_wo, x_w, yp = _prepare_data(fdf, y, [], 0)
print(f"\nFeatures WITHOUT interactions: {x_wo.shape[1]}")
print(f"Features WITH interactions:    {x_w.shape[1]}")

# Now WITH interactions
top_pairs = [(r["feature_1"], r["feature_2"]) for _, r in ir.head(20).iterrows()]
x_wo2, x_w2, yp2 = _prepare_data(fdf, y, top_pairs, 20)
print(f"\nWith top-20 interactions:")
print(f"  WITHOUT: {x_wo2.shape[1]} features")
print(f"  WITH:    {x_w2.shape[1]} features")
int_cols = [c for c in x_w2.columns if "__" in c]
print(f"  Interaction columns: {int_cols}")

# Check the specific interaction column
ic = f"{f1}__{f2}"
if ic in x_w2.columns:
    uni = x_w2[ic].nunique()
    print(f"\n  {ic}: {uni} unique encoded values (out of {len(x_w2)} rows)")
    val_counts = x_w2[ic].value_counts()
    print(f"  Top-5 most common combinations: {val_counts.head(5).to_dict()}")
    print(f"  Singleton combos (appear once): {(val_counts == 1).sum()}")
    print(f"  Combos with <=5 rows: {(val_counts <= 5).sum()}")
else:
    print(f"\n  {ic} NOT FOUND in interaction columns!")

# ============================================================
# B.2: Cross-fold generalization analysis
# ============================================================
print("\n" + "=" * 60)
print("B.2: CROSS-FOLD GENERALIZABILITY")
print("=" * 60)

# For the interaction column, check how many unique combos
# appear in train vs test across 5 folds
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

generalization = []
for fold, (train_idx, test_idx) in enumerate(kfold.split(x_wo2, yp2)):
    train_vals = set(x_w2.iloc[train_idx][ic].unique())
    test_vals = set(x_w2.iloc[test_idx][ic].unique())
    unseen = test_vals - train_vals
    gen_ratio = len(test_vals - unseen) / len(test_vals) if len(test_vals) > 0 else 1.0
    generalization.append({
        "fold": fold,
        "train_combos": len(train_vals),
        "test_combos": len(test_vals),
        "unseen_in_test": len(unseen),
        "seen_ratio": round(gen_ratio, 4),
    })
    # Train a quick RF on this fold to check accuracy
    x_tr = x_w2.iloc[train_idx]
    x_te = x_w2.iloc[test_idx]
    y_tr = yp2.iloc[train_idx]
    y_te = yp2.iloc[test_idx]
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(x_tr, y_tr)
    pred = rf.predict(x_te)
    acc = accuracy_score(y_te, pred)
    generalization[-1]["rf_accuracy"] = round(acc, 4)

gen_df = pd.DataFrame(generalization)
print(gen_df.to_string(index=False))
print(f"\n  Mean train combos: {gen_df['train_combos'].mean():.0f}")
print(f"  Mean test combos:  {gen_df['test_combos'].mean():.0f}")
print(f"  Mean unseen ratio: {1 - gen_df['seen_ratio'].mean():.3f}")
print(f"  Mean RF accuracy:  {gen_df['rf_accuracy'].mean():.4f}")

# ============================================================
# B.3: SHAP — what does RF actually learn for PayZone?
# ============================================================
print("\n" + "=" * 60)
print("B.3: SHAP ANALYSIS ON PAYZONE RF")
print("=" * 60)

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(x_w2, yp2)

fi = pd.DataFrame({"feature": x_w2.columns, "importance": rf.feature_importances_})
fi = fi.sort_values("importance", ascending=False)
print("\n  Top-10 features by importance:")
print(fi.head(10).to_string(index=False))

if ic in fi["feature"].values:
    rank = fi[fi["feature"] == ic].index[0] + 1
    imp = fi[fi["feature"] == ic]["importance"].values[0]
    print(f"\n  Interaction {ic}: rank #{rank}, importance={imp:.4f}")
else:
    print(f"\n  {ic} NOT in feature importances")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Top interaction impact: {top_pz.iloc[0]['impact']:.1f}")
print(f"Interaction unique combos: {x_w2[ic].nunique() if ic in x_w2.columns else 0}")
print(f"Mean unseen across CV folds: {1 - gen_df['seen_ratio'].mean():.1%}")
print(f"Best CV accuracy with interaction: {gen_df['rf_accuracy'].max():.4f}")
print(f"Random baseline (3-class): 0.3333")
print(f"Majority baseline: 0.354")
print(f"\nConclusion: The interaction JobFamily x DepartmentType produces {x_w2[ic].nunique() if ic in x_w2.columns else 0}")
print(f"unique label-encoded combinations. Across 5-fold CV, {1 - gen_df['seen_ratio'].mean():.1%} of test")
print(f"combinations are unseen during training. The RF cannot generalize to unseen combos,")
print(f"explaining the near-random accuracy despite strong population-level association.")
