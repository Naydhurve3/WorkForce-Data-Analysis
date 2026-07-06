"""Extract all findings from the v2.0 pipeline into research_paper/ folder."""
import json, os
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "research_paper" / "01_findings"
DATA = ROOT / "data" / "interaction"

os.makedirs(OUT, exist_ok=True)

# ── 1. Interaction Rankings ────────────────────────────────────────────────
def extract_interaction_rankings():
    src = DATA / "interaction_top50.json"
    if not src.exists():
        print("[SKIP] interaction_top50.json not found"); return
    with open(src) as f:
        data = json.load(f)
    rows = []
    for d in data:
        rows.append({
            "rank": d.get("rank", ""),
            "feature_1": d.get("feature_1", ""),
            "feature_2": d.get("feature_2", ""),
            "outcome": d.get("outcome", ""),
            "impact": d.get("impact", ""),
            "mutual_info": d.get("mutual_info", ""),
            "method": d.get("method", ""),
            "p_value": d.get("p_value", ""),
            "statistic": d.get("statistic", ""),
            "significant": d.get("significant", ""),
            "bonferroni": d.get("bonferroni", ""),
            "f1_type": d.get("f1_type", ""),
            "f2_type": d.get("f2_type", ""),
            "outcome_type": d.get("outcome_type", ""),
            "n": d.get("n", ""),
        })
    pd.DataFrame(rows).to_csv(OUT / "interaction_rankings.csv", index=False)
    print(f"[OK] interaction_rankings.csv — {len(rows)} rows")


# ── 2. Model Metrics ───────────────────────────────────────────────────────
def extract_model_metrics():
    src = DATA / "model_results.json"
    if not src.exists():
        print("[SKIP] model_results.json not found"); return
    with open(src) as f:
        data = json.load(f)
    rows = []
    for outcome, models in data.items():
        for model_key, model_data in models.items():
            mc = model_data.get("metrics_cv", {})
            m = model_data.get("metrics", {})
            rows.append({
                "outcome": outcome,
                "model": model_key,
                "accuracy_mean": mc.get("accuracy_mean", m.get("accuracy")),
                "accuracy_std": mc.get("accuracy_std", 0),
                "f1_mean": mc.get("f1_mean", m.get("f1")),
                "f1_std": mc.get("f1_std", 0),
                "roc_auc_mean": mc.get("roc_auc_mean", m.get("roc_auc")),
                "roc_auc_std": mc.get("roc_auc_std", 0),
                "has_interactions": "_with" in model_key,
            })
    pd.DataFrame(rows).to_csv(OUT / "model_metrics.csv", index=False)
    print(f"[OK] model_metrics.csv — {len(rows)} rows")


# ── 3. Deep Dive Stats ─────────────────────────────────────────────────────
def extract_deep_dives():
    src = DATA / "deep_dives.json"
    if not src.exists():
        print("[SKIP] deep_dives.json not found"); return
    with open(src) as f:
        data = json.load(f)
    rows = []
    for d in data:
        rows.append({
            "feature_1": d.get("f1"),
            "feature_2": d.get("f2"),
            "f1_value": d.get("f1_val"),
            "f2_value": d.get("f2_val"),
            "outcome": d.get("outcome"),
            "segment_size": d.get("n_segment"),
            "rest_size": d.get("n_rest"),
            "segment_mean": d.get("segment_outcome_mean"),
            "rest_mean": d.get("rest_outcome_mean"),
            "difference": d.get("difference"),
            "effect_size": d.get("effect_size"),
            "p_value": d.get("p_value"),
        })
    pd.DataFrame(rows).to_csv(OUT / "deep_dive_stats.csv", index=False)
    print(f"[OK] deep_dive_stats.csv — {len(rows)} rows")


# ── 4. Dataset Profile ─────────────────────────────────────────────────────
def extract_dataset_profile():
    src = ROOT / "data" / "raw" / "employee_data.csv"
    if not src.exists():
        print("[SKIP] employee_data.csv not found"); return
    df = pd.read_csv(src)
    info = []
    for col in df.columns:
        nulls = df[col].isna().sum()
        info.append({
            "column": col,
            "dtype": str(df[col].dtype),
            "non_null": int(df[col].notna().sum()),
            "null_count": int(nulls),
            "null_pct": round(nulls / len(df) * 100, 2),
            "unique": int(df[col].nunique()),
            "sample_values": str(df[col].dropna().unique()[:5].tolist()),
        })
    pd.DataFrame(info).to_csv(OUT / "dataset_profile.csv", index=False)
    print(f"[OK] dataset_profile.csv — {len(info)} columns")


# ── 5. Interaction Summary ─────────────────────────────────────────────────
def extract_interaction_summary():
    src = DATA / "interaction_summary.json"
    if not src.exists():
        print("[SKIP] interaction_summary.json not found"); return
    with open(src) as f:
        data = json.load(f)
    with open(OUT / "interaction_summary.txt", "w") as f:
        f.write("INTERACTION MINING SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total tests: {data.get('total_tests')}\n")
        f.write(f"Significant (p<0.05): {data.get('significant_p05')}\n")
        f.write(f"Significant (Bonferroni): {data.get('significant_bonferroni')}\n")
        f.write(f"Outcomes: {data.get('outcomes')}\n")
        f.write(f"Methods: {data.get('methods_used')}\n")
        top = data.get("top_pair", {})
        f.write(f"\nTop pair: {top.get('feature_1')} x {top.get('feature_2')}"
                f" -> {top.get('outcome')} (impact={top.get('impact')})\n")
    print("[OK] interaction_summary.txt")


# ── 6. Top 10 Discoveries (Narrative) ──────────────────────────────────────
def extract_top_discoveries():
    src = DATA / "interaction_top50.json"
    if not src.exists():
        print("[SKIP] interaction_top50.json not found"); return
    with open(src) as f:
        data = json.load(f)

    outcome_labels = {
        "PayZone_encoded": "Pay Zone (Compensation)",
        "is_minority_dept": "Minority Department Status",
        "SeniorityLevel": "Seniority Level",
        "PerfScore": "Performance Score",
        "is_terminated": "Termination (Attrition)",
    }
    method_labels = {"chi2": "chi2", "anova": "ANOVA", "pearson": "Pearson r"}

    lines = ["TOP 10 INTERACTION DISCOVERIES\n", "=" * 50, ""]
    for i, d in enumerate(data[:10], 1):
        outcome_label = outcome_labels.get(d["outcome"], d["outcome"])
        method_lbl = method_labels.get(d["method"], d["method"])
        lines.append(f"{i}. {d['feature_1']} x {d['feature_2']} -> {outcome_label}")
        lines.append(f"   Impact Score: {d['impact']}")
        lines.append(f"   Method: {method_lbl} | p-value: {d['p_value']:.2e}")
        lines.append(f"   Mutual Information: {d['mutual_info']}")
        lines.append(f"   Bonferroni significant: {d['bonferroni']}")
        lines.append("")

    with open(OUT / "top_10_discoveries.txt", "w") as f:
        f.write("\n".join(lines))

    # Also CSV
    rows = []
    for d in data[:10]:
        rows.append({
            "rank": d.get("rank"),
            "feature_1": d.get("feature_1"),
            "feature_2": d.get("feature_2"),
            "outcome": outcome_labels.get(d.get("outcome"), d.get("outcome")),
            "impact": d.get("impact"),
            "method": method_labels.get(d.get("method"), d.get("method")),
            "p_value": f"{d.get('p_value'):.2e}",
            "mutual_info": d.get("mutual_info"),
        })
    pd.DataFrame(rows).to_csv(OUT / "top_10_discoveries.csv", index=False)
    print("[OK] top_10_discoveries.txt + top_10_discoveries.csv")


# ── 7. Model Comparison Summary ────────────────────────────────────────────
def extract_model_summary():
    src = DATA / "model_results.json"
    if not src.exists():
        print("[SKIP] model_results.json not found"); return
    with open(src) as f:
        data = json.load(f)

    outcome_labels = {
        "is_terminated": "Termination",
        "PerfScore": "Performance",
        "PayZone_encoded": "Pay Zone",
        "is_minority_dept": "Minority Dept",
        "SeniorityLevel": "Seniority",
    }

    with open(OUT / "model_summary.txt", "w") as f:
        f.write("MODEL COMPARISON SUMMARY (5-fold CV)\n")
        f.write("=" * 60 + "\n\n")
        for outcome, models in data.items():
            label = outcome_labels.get(outcome, outcome)
            f.write(f"Outcome: {label}\n{'-' * 40}\n")
            for model_key, model_data in models.items():
                mc = model_data.get("metrics_cv", {})
                m = model_data.get("metrics", {})
                version = "WITH interactions" if "_with" in model_key else "WITHOUT interactions"
                acc_m = mc.get("accuracy_mean", m.get("accuracy"))
                acc_s = mc.get("accuracy_std", 0)
                f1_m = mc.get("f1_mean", m.get("f1"))
                f1_s = mc.get("f1_std", 0)
                auc_m = mc.get("roc_auc_mean", m.get("roc_auc"))
                auc_s = mc.get("roc_auc_std", 0)
                acc_str = f"{acc_m:.3f}\u00b1{acc_s:.3f}" if acc_m is not None else "N/A"
                f1_str = f"{f1_m:.3f}\u00b1{f1_s:.3f}" if f1_m is not None else "N/A"
                auc_str = f"{auc_m:.3f}\u00b1{auc_s:.3f}" if auc_m is not None else "N/A"
                f.write(f"  {model_key.replace('_with', ' +interactions').replace('_without', '')}:  "
                        f"Acc={acc_str}  F1={f1_str}  AUC={auc_str}  [{version}]\n")
            f.write("\n")
    print("[OK] model_summary.txt")


# ── 8. Feature Engineering Summary ─────────────────────────────────────────
def extract_feature_summary():
    src = DATA / "feature_summary.json"
    if not src.exists():
        print("[SKIP] feature_summary.json not found"); return
    with open(src) as f:
        data = json.load(f)
    rows = []
    for item in data:
        if isinstance(item, dict):
            rows.append({
                "feature": item.get("feature", ""),
                "type": item.get("type", item.get("dtype", "")),
                "n_unique": item.get("unique", ""),
                "mean": item.get("mean", ""),
                "std": item.get("std", ""),
            })
    pd.DataFrame(rows).to_csv(OUT / "feature_engineering_summary.csv", index=False)
    print(f"[OK] feature_engineering_summary.csv — {len(rows)} features")


# ── 9. Model Feature Importance ────────────────────────────────────────────
def extract_feature_importance():
    src = DATA / "model_results.json"
    if not src.exists():
        print("[SKIP] model_results.json not found"); return
    with open(src) as f:
        data = json.load(f)
    rows = []
    for outcome, models in data.items():
        for model_key, model_data in models.items():
            fi = model_data.get("feature_importance", {})
            if not fi:
                continue
            for feat, imp in fi.items():
                rows.append({
                    "outcome": outcome,
                    "model": model_key,
                    "feature": feat,
                    "importance": imp,
                })
    pd.DataFrame(rows).to_csv(OUT / "feature_importance.csv", index=False)
    print(f"[OK] feature_importance.csv — {len(rows)} rows")


# ── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path
    print("Extracting research findings...\n")
    extract_interaction_rankings()
    extract_model_metrics()
    extract_deep_dives()
    extract_dataset_profile()
    extract_interaction_summary()
    extract_top_discoveries()
    extract_model_summary()
    extract_feature_summary()
    extract_feature_importance()
    print(f"\nAll outputs saved to: {OUT}")
