"""Phase A.3: Paired t-test + Wilcoxon signed-rank on per-fold CV metrics.
   Compares WITH vs WITHOUT interactions for each outcome+model combo."""
import json, numpy as np
from scipy.stats import ttest_rel, wilcoxon
from pathlib import Path

DATA = Path("data") / "interaction"
OUT = DATA / "statistical_tests.json"
MR = DATA / "model_results.json"

with open(MR) as f:
    data = json.load(f)

results = {}
for oname, odata in data.items():
    outcome_results = {}
    # Find all model types (LR, RF, XGB)
    models = set(k.rsplit("_", 1)[0] for k in odata.keys())
    for mname in sorted(models):
        wo_key = f"{mname}_without"
        w_key = f"{mname}_with"
        if wo_key not in odata or w_key not in odata:
            continue
        wo = odata[wo_key].get("metrics_cv", {})
        w = odata[w_key].get("metrics_cv", {})

        pair_results = {}
        for metric in ["accuracy", "f1", "roc_auc", "balanced_accuracy", "mcc", "pr_auc"]:
            wo_folds = wo.get(f"{metric}_folds")
            w_folds = w.get(f"{metric}_folds")
            if not wo_folds or not w_folds or len(wo_folds) < 2:
                continue
            a, b = np.array(wo_folds), np.array(w_folds)
            diffs = b - a
            t_stat, t_p = ttest_rel(a, b)
            try:
                w_stat, w_p = wilcoxon(a, b, zero_method="wilcox")
            except Exception:
                w_stat, w_p = None, 1.0
            cohens_d = diffs.mean() / (diffs.std(ddof=1) + 1e-10)
            mean_wo, mean_w = float(a.mean()), float(b.mean())
            pair_results[metric] = {
                "without_mean": round(mean_wo, 4),
                "with_mean": round(mean_w, 4),
                "delta": round(float(diffs.mean()), 4),
                "delta_std": round(float(diffs.std(ddof=1)), 4),
                "t_statistic": round(float(t_stat), 4),
                "t_pvalue": round(float(t_p), 4),
                "wilcoxon_statistic": round(float(w_stat), 4) if w_stat is not None else None,
                "wilcoxon_pvalue": round(float(w_p), 4),
                "cohens_d": round(float(cohens_d), 4),
                "significant_t05": bool(t_p < 0.05),
                "significant_w05": bool(w_p < 0.05),
                "n_folds": len(wo_folds),
            }
        if pair_results:
            outcome_results[mname] = pair_results
    if outcome_results:
        results[oname] = outcome_results

with open(OUT, "w") as f:
    json.dump(results, f, indent=2, default=str)

print(f"Statistical tests saved to {OUT}")
print()
for oname, odata in results.items():
    for mname, mdata in odata.items():
        acc = mdata.get("accuracy", {})
        if acc:
            sig_t = "p<0.05" if acc.get("significant_t05") else "n.s."
            sig_w = "p<0.05" if acc.get("significant_w05") else "n.s."
            print(f"  {oname:20s} {mname:3s} acc: d={acc['delta']:+.4f} t-test={sig_t} wilcoxon={sig_w} cohens_d={acc['cohens_d']:+.3f}")
