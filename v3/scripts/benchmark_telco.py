"""
Benchmark: Run interaction mining pipeline on Telco Customer Churn dataset.
"""
import sys, json, warnings, time
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.base import clone
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from scipy.stats import chi2_contingency, f_oneway
import xgboost as xgb

RESULTS_DIR = Path("data") / "benchmark_telco"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = Path("data") / "raw" / "telco"
RAW_DIR.mkdir(parents=True, exist_ok=True)
logger = lambda msg: print(f"[TELCO] {msg}")


def load_telco_data():
    csv_path = RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        logger(f"Loaded {len(df)} rows from local file")
    else:
        logger("Downloading Telco Customer Churn via kagglehub...")
        import kagglehub
        download_path = kagglehub.dataset_download("blastchar/telco-customer-churn")
        import shutil
        for f in Path(download_path).iterdir():
            if f.suffix == ".csv":
                shutil.copy2(f, csv_path)
                break
        df = pd.read_csv(csv_path)
        logger(f"Loaded {len(df)} rows after download")
    df["churn"] = (df["Churn"] == "Yes").astype(int)
    df = df.drop(columns=["Churn", "customerID"])
    return df


def engineer_features(df):
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    for c in df.select_dtypes(include="object").columns:
        df[c] = LabelEncoder().fit_transform(df[c].astype(str))
    return df


def run_interaction_mining_fast(df, outcome_col="churn"):
    logger("Running interaction mining...")
    y = df[outcome_col]
    X = df.drop(columns=[outcome_col])
    feature_cols = [c for c in X.columns if X[c].dtype in [np.int64, np.float64, int, float]]
    logger(f"Mining {len(feature_cols)} features")

    def _is_cat(c):
        return X[c].nunique() <= 15

    n = len(df)
    results = []
    cols = feature_cols
    total_pairs = len(cols) * (len(cols) - 1) // 2
    pair_idx = 0
    n_bonferroni = total_pairs
    t0 = time.time()
    for i, f1 in enumerate(cols):
        for f2 in cols[i + 1:]:
            pair_idx += 1
            v1, v2 = X[f1].values, X[f2].values
            c1, c2 = _is_cat(f1), _is_cat(f2)

            try:
                if c1 and c2:
                    ct = pd.crosstab(X[f1], X[f2])
                    stat, p, _, _ = chi2_contingency(ct)
                    method = "chi2"
                elif not c1 and not c2:
                    from scipy.stats import pearsonr
                    r_val, p = pearsonr(v1, v2)
                    stat = abs(r_val) * 100
                    method = "pearson"
                else:
                    num_v = v1 if not c1 else v2
                    cat_v = v2 if not c1 else v1
                    groups = [num_v[cat_v == c] for c in np.unique(cat_v) if np.sum(cat_v == c) > 1]
                    if len(groups) >= 2:
                        stat, p = f_oneway(*groups)
                    else:
                        stat, p = 0, 1.0
                    method = "anova"
            except Exception:
                continue

            try:
                from sklearn.feature_selection import mutual_info_classif
                x_vals = pd.DataFrame(X[f1].astype(float).fillna(X[f1].median()))
                y_vals = y.astype(int)
                mi = float(mutual_info_classif(x_vals, y_vals, random_state=42)[0])
            except Exception:
                mi = 0.0

            if method == "anova":
                num_v = v1 if not c1 else v2
                cat_v = v2 if not c1 else v1
                grand_mean = np.mean(num_v)
                ss_total = np.sum((num_v - grand_mean) ** 2)
                ss_between = sum(
                    np.sum(num_v[cat_v == c] - grand_mean) ** 2 / max(np.sum(cat_v == c), 1)
                    for c in np.unique(cat_v) if np.sum(cat_v == c) > 1
                )
                eff = ss_between / ss_total if ss_total > 0 else mi
            elif method == "pearson":
                eff = abs(r_val) ** 2 if not np.isnan(abs(r_val)) else mi
            else:
                eff = mi

            p_val = float(p) if float(p) > 0 else 1e-300
            impact = round(-np.log10(max(p_val, 1e-300)) * max(eff, 0.001), 4)

            results.append({
                "feature_1": f1, "feature_2": f2,
                "method": method,
                "p_value": round(p_val, 6),
                "statistic": round(float(stat), 4),
                "impact": impact,
                "mutual_info": round(float(mi), 4),
                "outcome_type": outcome_col,
                "significant": bool(p_val < 0.05),
                "bonferroni": bool(p_val * n_bonferroni < 0.05),
            })

    logger(f"Tested {pair_idx} pairs in {time.time()-t0:.1f}s")
    rdf = pd.DataFrame(results).sort_values("impact", ascending=False).reset_index(drop=True)
    rdf["rank"] = range(1, len(rdf) + 1)
    rdf.to_csv(RESULTS_DIR / "telco_interaction_rankings.csv", index=False)
    if len(rdf):
        t = rdf.iloc[0]
        logger(f"Top: {t['feature_1']} x {t['feature_2']} (impact={t['impact']:.4f}, p={t['p_value']:.2e})")
    return rdf


def run_models(df, outcome_col="churn"):
    logger("Running predictive models with 5-fold CV...")
    int_path = RESULTS_DIR / "telco_interaction_rankings.csv"
    top_ints = pd.read_csv(int_path).head(15) if int_path.exists() else pd.DataFrame()

    X = df.drop(columns=[outcome_col])
    y = df[outcome_col]
    feature_cols = [c for c in X.columns if X[c].dtype in [np.int64, np.float64, int, float]]
    X = X[feature_cols].astype(float)

    X_int = X.copy()
    for _, row in top_ints.iterrows():
        f1, f2 = row["feature_1"], row["feature_2"]
        if f1 in X.columns and f2 in X.columns:
            f1_num = X[f1].nunique() > 5
            f2_num = X[f2].nunique() > 5
            if f1_num and f2_num:
                X_int[f"{f1}__{f2}"] = (X[f1] * X[f2]).fillna(0)
            else:
                X_int[f"{f1}__{f2}"] = LabelEncoder().fit_transform(
                    (X[f1].astype(str) + "_x_" + X[f2].astype(str)).values
                )

    models_def = {
        "LR": LogisticRegression(max_iter=2000, random_state=42, class_weight="balanced"),
        "RF": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, class_weight="balanced"),
        "XGB": xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  subsample=0.8, random_state=42, eval_metric="logloss", verbosity=0),
    }

    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []

    for name, base_model in models_def.items():
        for variant, X_data in [("without", X), ("with", X_int)]:
            fold_accs, fold_f1s, fold_aucs = [], [], []

            for train_idx, test_idx in kfold.split(X, y):
                x_tr = X_data.iloc[train_idx]
                x_te = X_data.iloc[test_idx]
                y_tr = y.iloc[train_idx]
                y_te = y.iloc[test_idx]

                try:
                    fold_model = clone(base_model)
                    fold_model.fit(x_tr, y_tr)
                    y_pred = fold_model.predict(x_te)
                    y_proba = fold_model.predict_proba(x_te) if hasattr(fold_model, "predict_proba") else None
                    fold_accs.append(float(accuracy_score(y_te, y_pred)))
                    fold_f1s.append(float(f1_score(y_te, y_pred)))
                    if y_proba is not None and y_proba.ndim == 2 and y_proba.shape[1] == 2:
                        fold_aucs.append(float(roc_auc_score(y_te, y_proba[:, 1])))
                except Exception as e:
                    logger(f"  {name} ({variant}) fold FAILED: {e}")

            if not fold_accs:
                continue

            acc_mean = round(float(np.mean(fold_accs)), 4)
            acc_std = round(float(np.std(fold_accs)), 4)
            f1_mean = round(float(np.mean(fold_f1s)), 4)
            f1_std = round(float(np.std(fold_f1s)), 4)
            auc_mean = round(float(np.mean(fold_aucs)), 4) if fold_aucs else None
            auc_std = round(float(np.std(fold_aucs)), 4) if fold_aucs else None

            rows.append({
                "model": name, "variant": variant,
                "accuracy_mean": acc_mean, "accuracy_std": acc_std,
                "f1_mean": f1_mean, "f1_std": f1_std,
                "roc_auc_mean": auc_mean, "roc_auc_std": auc_std,
            })
            logger(f"  {name:4s} ({variant:7s}): acc={acc_mean:.4f}+-{acc_std:.4f} f1={f1_mean:.4f}+-{f1_std:.4f}" +
                   (f" auc={auc_mean:.4f}+-{auc_std:.4f}" if auc_mean else ""))

    rdf = pd.DataFrame(rows)
    rdf.to_csv(RESULTS_DIR / "telco_model_metrics.csv", index=False)
    return rdf


def main():
    logger("=" * 60)
    logger("Telco Customer Churn Benchmark — Interaction Mining")
    logger("=" * 60)

    df = load_telco_data()
    df = engineer_features(df)
    logger(f"Engineered: {df.shape[1]} total cols, target mean={df['churn'].mean():.3f}")

    int_results = run_interaction_mining_fast(df)
    model_results = run_models(df)

    top5 = int_results.head(5)
    print("\n" + "=" * 60)
    print("TOP 5 TELCO INTERACTIONS")
    print("=" * 60)
    for _, r in top5.iterrows():
        print(f"  {r['feature_1']:30s} x {r['feature_2']:30s}  impact={r['impact']:8.2f}  p={r['p_value']:.2e}")

    print("\n" + "=" * 60)
    print("MODEL COMPARISON (TELCO)")
    print("=" * 60)
    for _, r in model_results[model_results["variant"].isin(["without", "with"])].iterrows():
        std_s = f"+-{r['accuracy_std']:.4f}" if r['accuracy_std'] > 0 else ""
        print(f"  {r['model']:4s} ({r['variant']:7s}): acc={r['accuracy_mean']:.4f}{std_s} f1={r['f1_mean']:.4f}" +
              (f" auc={r['roc_auc_mean']:.4f}" if r['roc_auc_mean'] else ""))

    summary = {
        "dataset": "Telco Customer Churn (Kaggle)",
        "n_rows": len(df),
        "n_features_engineered": df.shape[1] - 1,
        "n_interactions_tested": len(int_results),
        "top_interaction": {
            "feature_1": str(int_results.iloc[0]["feature_1"]),
            "feature_2": str(int_results.iloc[0]["feature_2"]),
            "impact": float(int_results.iloc[0]["impact"]),
            "p_value": float(int_results.iloc[0]["p_value"]),
        },
        "best_acc_without": float(model_results.loc[model_results["variant"] == "without", "accuracy_mean"].max()),
        "best_acc_with": float(model_results.loc[model_results["variant"] == "with", "accuracy_mean"].max()),
    }
    with open(RESULTS_DIR / "telco_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
