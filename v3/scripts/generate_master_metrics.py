"""Phase 0.1: Generate master metrics spreadsheet from model_results.json.
   Single source of truth for every number in the paper."""
import json, csv, sys
from pathlib import Path

DATA = Path("data") / "interaction"
OUT = DATA / "master_metrics.csv"
MR = DATA / "model_results.json"

if not MR.exists():
    print(f"ERROR: {MR} not found. Run phase4 first.")
    sys.exit(1)

with open(MR) as f:
    data = json.load(f)

rows = []
for oname in sorted(data.keys()):
    for key in sorted(data[oname].keys()):
        res = data[oname][key]
        m = res.get("metrics", {})
        mc = res.get("metrics_cv", {})
        cm = res.get("cm") or m.get("cm", [])

        variant = "with_int" if "_with" in key and "_without" not in key else "without_int"
        model_type = key.split("_")[0]
        outcome_label = oname.replace("_encoded", "")

        row = {
            "outcome": outcome_label,
            "model": model_type,
            "variant": variant,
            "n_configs": 1,
            # basic metrics
            "accuracy": m.get("accuracy", ""),
            "balanced_accuracy": m.get("balanced_accuracy", ""),
            "f1": m.get("f1", ""),
            "precision": m.get("precision", ""),
            "recall": m.get("recall", ""),
            "mcc": m.get("mcc", ""),
            "roc_auc": m.get("roc_auc", ""),
            "pr_auc": m.get("pr_auc", ""),
            "brier": m.get("brier", ""),
            "ece": m.get("ece", ""),
            # CV means
            "accuracy_cv": mc.get("accuracy_mean", ""),
            "accuracy_std": mc.get("accuracy_std", ""),
            "balanced_accuracy_cv": mc.get("balanced_accuracy_mean", ""),
            "f1_cv": mc.get("f1_mean", ""),
            "f1_std": mc.get("f1_std", ""),
            "precision_cv": mc.get("precision_mean", ""),
            "recall_cv": mc.get("recall_mean", ""),
            "mcc_cv": mc.get("mcc_mean", ""),
            "roc_auc_cv": mc.get("roc_auc_mean", ""),
            "roc_auc_std": mc.get("roc_auc_std", ""),
            "pr_auc_cv": mc.get("pr_auc_mean", ""),
            "brier_cv": mc.get("brier_mean", ""),
            "ece_cv": mc.get("ece_mean", ""),
            # per-fold
            "accuracy_folds": mc.get("accuracy_folds", []),
            "roc_auc_folds": mc.get("roc_auc_folds", []),
            # confusion matrix total
            "cm_total": sum(sum(r) for r in cm) if cm else "",
        }
        rows.append(row)

if not rows:
    print("ERROR: no rows generated")
    sys.exit(1)

with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader()
    w.writerows(rows)

print(f"Master metrics: {OUT} ({len(rows)} configs)")
for r in rows:
    print(f"  {r['outcome']:20s} {r['model']:3s} {r['variant']:12s} acc={r['accuracy_cv']} f1={r['f1_cv']} auc={r['roc_auc_cv']}")
