"""Phase C: Robustness Analysis.
Multi-seed stability, noise injection, missing data, subsampling.
"""
import sys, json, warnings, copy
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from loguru import logger
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.models import _prepare_data, _encode_interaction_pair, _eval_model, _LEAKAGE_EXCLUDE

warnings.filterwarnings("ignore")
logger.remove()
logger.add(sys.stdout, format="{message}")

OUT = Path("data") / "interaction"
cfg = InteractionConfig()

# ── Load data ──────────────────────────────────────────────────────────────
raw = DataLoader.load(cfg.raw_path, validate=False)
eng = FeatureEngineer(cfg)
fdf = eng.compute_all(raw)

ir = pd.read_parquet(OUT / "interaction_results.parquet")
top20 = [(r["feature_1"], r["feature_2"]) for _, r in ir.head(20).iterrows()]

OUTCOMES = {
    "is_terminated": raw["EmployeeStatus"].str.lower().str.contains("terminat").astype(int),
    "PerfScore": fdf.get("PerfScore", raw["Current Employee Rating"].fillna(0).astype(int)),
    "PayZone_encoded": raw["PayZone"].map({"Zone A": 0, "Zone B": 1, "Zone C": 2}).fillna(0).astype(int),
    "SeniorityLevel": fdf.get("SeniorityLevel", pd.Series(1, index=raw.index)),
}
_is_minority_dept = pd.Series(0, index=raw.index)
try:
    for dept in raw["DepartmentType"].unique():
        dept_mask = raw["DepartmentType"] == dept
        dept_data = raw[dept_mask]
        majority = dept_data["GenderCode"].value_counts().index[0]
        _is_minority_dept.loc[dept_mask] = (dept_data["GenderCode"] != majority).astype(int)
except Exception:
    pass
_is_minority_dept = _is_minority_dept.fillna(0).astype(int)
OUTCOMES["is_minority_dept"] = _is_minority_dept

MODELS = {
    "LR": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
    "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
}
try:
    MODELS["XGB"] = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42, eval_metric="logloss", verbosity=0)
except Exception:
    pass

N_FOLDS = 5
N_SEEDS = 10

# ═══════════════════════════════════════════════════════════════════════════
# C.1 — Multi-seed stability
# ═══════════════════════════════════════════════════════════════════════════
logger.info("=" * 60)
logger.info("C.1: MULTI-SEED STABILITY (n={})".format(N_SEEDS))
logger.info("=" * 60)

seed_records = []
for seed in range(N_SEEDS):
    kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for oname, oseries in OUTCOMES.items():
        x_wo, x_w, y = _prepare_data(fdf, oseries, top20, 20, oname)
        for mname, base_model in MODELS.items():
            for variant, xdata in [("without", x_wo), ("with", x_w)]:
                if variant == "with" and x_w.shape[1] == x_wo.shape[1]:
                    continue
                fold_accs, fold_f1s, fold_aucs = [], [], []
                for train_idx, test_idx in kf.split(xdata, y):
                    try:
                        m = clone(base_model)
                        m.set_params(random_state=seed)
                        m.fit(xdata.iloc[train_idx], y.iloc[train_idx])
                        pred = m.predict(xdata.iloc[test_idx])
                        fold_accs.append(accuracy_score(y.iloc[test_idx], pred))
                        fold_f1s.append(f1_score(y.iloc[test_idx], pred, average="weighted"))
                        if hasattr(m, "predict_proba"):
                            proba = m.predict_proba(xdata.iloc[test_idx])
                            if proba.shape[1] == 2:
                                fold_aucs.append(roc_auc_score(y.iloc[test_idx], proba[:, 1]))
                    except Exception:
                        pass
                if fold_accs:
                    seed_records.append({
                        "seed": seed, "outcome": oname, "model": mname, "variant": variant,
                        "acc_mean": float(np.mean(fold_accs)), "acc_std": float(np.std(fold_accs)),
                        "f1_mean": float(np.mean(fold_f1s)), "f1_std": float(np.std(fold_f1s)),
                        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else None,
                    })

seed_df = pd.DataFrame(seed_records)
seed_df.to_csv(OUT / "robustness_multiseed.csv", index=False)

# Summarize: within-config stability across seeds
summary_rows = []
for (oname, mname, variant), grp in seed_df.groupby(["outcome", "model", "variant"]):
    accs = grp["acc_mean"].values
    summary_rows.append({
        "outcome": oname, "model": mname, "variant": variant,
        "acc_overall_mean": round(float(np.mean(accs)), 4),
        "acc_overall_std": round(float(np.std(accs)), 4),
        "acc_min": round(float(np.min(accs)), 4),
        "acc_max": round(float(np.max(accs)), 4),
        "acc_range": round(float(np.max(accs) - np.min(accs)), 4),
    })
    logger.info("  {:<18s} {:<4s} {:<7s}  acc={:.4f}±{:.4f}  range={:.4f}".format(
        oname, mname, variant,
        summary_rows[-1]["acc_overall_mean"],
        summary_rows[-1]["acc_overall_std"],
        summary_rows[-1]["acc_range"],
    ))

multi_seed_summary = pd.DataFrame(summary_rows)
multi_seed_summary.to_csv(OUT / "robustness_multiseed_summary.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# C.2 — Noise injection
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("C.2: NOISE INJECTION")
logger.info("=" * 60)

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
noise_records = []

oname = "is_terminated"
oseries = OUTCOMES[oname]
x_wo, x_w, y = _prepare_data(fdf, oseries, top20, 20, oname)
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

for noise_std in NOISE_LEVELS:
    x_noisy = x_w.copy().astype(float)
    if noise_std > 0:
        for c in x_noisy.columns:
            c_std = x_noisy[c].std()
            if c_std > 0 and "__" not in c:
                x_noisy[c] += np.random.RandomState(42).normal(0, noise_std * c_std, size=len(x_noisy))

    for mname, base_model in MODELS.items():
        if mname not in ["RF", "XGB"]:
            continue
        fold_accs, fold_f1s, fold_aucs = [], [], []
        for train_idx, test_idx in kf.split(x_noisy, y):
            try:
                m = clone(base_model)
                m.fit(x_noisy.iloc[train_idx], y.iloc[train_idx])
                pred = m.predict(x_noisy.iloc[test_idx])
                fold_accs.append(accuracy_score(y.iloc[test_idx], pred))
                fold_f1s.append(f1_score(y.iloc[test_idx], pred))
                proba = m.predict_proba(x_noisy.iloc[test_idx])
                if proba.shape[1] == 2:
                    fold_aucs.append(roc_auc_score(y.iloc[test_idx], proba[:, 1]))
            except Exception:
                pass
        noise_records.append({
            "noise_std": noise_std, "model": mname,
            "acc": round(float(np.mean(fold_accs)), 4),
            "f1": round(float(np.mean(fold_f1s)), 4),
            "auc": round(float(np.mean(fold_aucs)), 4) if fold_aucs else None,
        })
        logger.info("  noise={:.2f} {:<4s}  acc={:.4f}  f1={:.4f}".format(
            noise_std, mname, noise_records[-1]["acc"], noise_records[-1]["f1"],
        ))

noise_df = pd.DataFrame(noise_records)
noise_df.to_csv(OUT / "robustness_noise.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# C.3 — Missing data robustness
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("C.3: MISSING DATA ROBUSTNESS")
logger.info("=" * 60)

MISSING_RATES = [0.0, 0.05, 0.10, 0.20]
missing_records = []

oname = "is_terminated"
oseries = OUTCOMES[oname]
x_wo, x_w, y = _prepare_data(fdf, oseries, top20, 20, oname)

rf_base = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced")

for rate in MISSING_RATES:
    fold_accs, fold_f1s = [], []
    for train_idx, test_idx in kf.split(x_w, y):
        try:
            x_train = x_w.iloc[train_idx].copy().astype(float)
            x_test = x_w.iloc[test_idx].copy().astype(float)
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]

            if rate > 0:
                rng = np.random.RandomState(42)
                mask_train = rng.binomial(1, rate, size=x_train.shape).astype(bool)
                mask_test = rng.binomial(1, rate, size=x_test.shape).astype(bool)
                x_train[mask_train] = np.nan
                x_test[mask_test] = np.nan
                x_train = x_train.fillna(x_train.median())
                x_test = x_test.fillna(x_train.median())

            m = clone(rf_base)
            m.fit(x_train, y_train)
            pred = m.predict(x_test)
            fold_accs.append(accuracy_score(y_test, pred))
            fold_f1s.append(f1_score(y_test, pred))
        except Exception:
            pass

    missing_records.append({
        "missing_rate": rate,
        "acc": round(float(np.mean(fold_accs)), 4),
        "f1": round(float(np.mean(fold_f1s)), 4),
    })
    logger.info("  missing={:.0%}  acc={:.4f}  f1={:.4f}".format(
        rate, missing_records[-1]["acc"], missing_records[-1]["f1"],
    ))

missing_df = pd.DataFrame(missing_records)
missing_df.to_csv(OUT / "robustness_missing.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# C.4 — Subsampling (learning curves)
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("C.4: SUBSAMPLING / LEARNING CURVES")
logger.info("=" * 60)

SUBSAMPLE_RATIOS = [0.25, 0.50, 0.75, 1.0]
subsample_records = []

oname = "is_terminated"
oseries = OUTCOMES[oname]
x_wo, x_w, y = _prepare_data(fdf, oseries, top20, 20, oname)

for ratio in SUBSAMPLE_RATIOS:
    fold_accs, fold_f1s = [], []
    for train_idx, test_idx in kf.split(x_w, y):
        try:
            y_train = y.iloc[train_idx]
            if ratio < 1.0:
                sub_rng = np.random.RandomState(42)
                n_sub = max(2, int(len(train_idx) * ratio))
                sub_idx = sub_rng.choice(train_idx, size=n_sub, replace=False)
                x_sub = x_w.loc[sub_idx]
                y_sub = y.loc[sub_idx]
            else:
                x_sub = x_w.iloc[train_idx]
                y_sub = y_train

            m = clone(rf_base)
            m.fit(x_sub, y_sub)
            pred = m.predict(x_w.iloc[test_idx])
            fold_accs.append(accuracy_score(y.iloc[test_idx], pred))
            fold_f1s.append(f1_score(y.iloc[test_idx], pred))
        except Exception:
            pass

    subsample_records.append({
        "sample_ratio": ratio,
        "acc": round(float(np.mean(fold_accs)), 4),
        "f1": round(float(np.mean(fold_f1s)), 4),
    })
    logger.info("  ratio={:.0%}  acc={:.4f}  f1={:.4f}".format(
        ratio, subsample_records[-1]["acc"], subsample_records[-1]["f1"],
    ))

subsample_df = pd.DataFrame(subsample_records)
subsample_df.to_csv(OUT / "robustness_subsample.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════════
# Final summary
# ═══════════════════════════════════════════════════════════════════════════
logger.info("\n" + "=" * 60)
logger.info("PHASE C COMPLETE")
logger.info("=" * 60)

summary = {
    "multi_seed": {
        "n_seeds": N_SEEDS,
        "n_configs": len(seed_records),
        "max_range": float(multi_seed_summary["acc_range"].max()),
        "mean_range": float(multi_seed_summary["acc_range"].mean()),
    },
    "noise": {
        "levels_tested": NOISE_LEVELS,
        "acc_drop_at_10pct": None,
    },
    "missing": {
        "rates_tested": MISSING_RATES,
        "acc_drop_at_20pct": None,
    },
    "subsample": {
        "ratios_tested": SUBSAMPLE_RATIOS,
    },
}

if len(noise_records) >= 2:
    base_acc = noise_records[0]["acc"]
    worst_acc = min(r["acc"] for r in noise_records)
    summary["noise"]["acc_drop_at_10pct"] = round(base_acc - worst_acc, 4)

if len(missing_records) >= 2:
    base_acc = missing_records[0]["acc"]
    worst_acc = min(r["acc"] for r in missing_records)
    summary["missing"]["acc_drop_at_20pct"] = round(base_acc - worst_acc, 4)

with open(OUT / "robustness_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

logger.info("Outputs:")
logger.info("  robustness_multiseed.csv")
logger.info("  robustness_multiseed_summary.csv")
logger.info("  robustness_noise.csv")
logger.info("  robustness_missing.csv")
logger.info("  robustness_subsample.csv")
logger.info("  robustness_summary.json")
