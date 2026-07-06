"""Confusion matrix analysis: per-class precision/recall/F1 from CV results."""
import json
import numpy as np
import pandas as pd
from pathlib import Path

DATA = Path("data") / "interaction"
OUT = DATA / "error_analysis.csv"

def analyze_errors():
    with open(DATA / "model_results.json") as f:
        data = json.load(f)

    rows = []
    for oname, odata in data.items():
        for key, res in odata.items():
            cm = res.get("cm") or res.get("metrics", {}).get("cm")
            if cm is None or not isinstance(cm, list) or len(cm) < 2:
                continue

            cm_arr = np.array(cm)
            n_classes = cm_arr.shape[0]

            for c in range(n_classes):
                tp = cm_arr[c, c]
                fp = cm_arr[:, c].sum() - tp
                fn = cm_arr[c, :].sum() - tp
                tn = cm_arr.sum() - tp - fp - fn

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
                fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                rows.append({
                    "outcome": oname,
                    "model": key,
                    "class": c,
                    "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                    "precision": round(float(precision), 4),
                    "recall": round(float(recall), 4),
                    "f1": round(float(f1), 4),
                    "fpr": round(float(fpr), 4),
                    "fnr": round(float(fnr), 4),
                    "accuracy": round(float((tp + tn) / cm_arr.sum()), 4),
                })

    pdf = pd.DataFrame(rows)
    pdf.to_csv(OUT, index=False)

    print("=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    for oname in pdf["outcome"].unique():
        print(f"\n--- {oname} ---")
        odf = pdf[pdf["outcome"] == oname]
        for _, r in odf.iterrows():
            print(f"  {r['model']:20s} class={r['class']}: P={r['precision']:.3f} R={r['recall']:.3f} "
                  f"F1={r['f1']:.3f} FPR={r['fpr']:.3f} FNR={r['fnr']:.3f}")

    print(f"\nOutput saved to {OUT}")
    return pdf

if __name__ == "__main__":
    analyze_errors()
