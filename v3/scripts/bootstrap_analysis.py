"""Bootstrap confidence intervals from CV fold data."""
import json
import numpy as np
from pathlib import Path

DATA = Path("data") / "interaction"
OUT = DATA / "bootstrap_cis.json"

def bootstrap_ci(values, n_resamples=1000, ci=0.95):
    values = np.array(values)
    if len(values) < 2:
        return {"mean": float(np.mean(values)), "ci_lower": float(np.mean(values)), "ci_upper": float(np.mean(values))}
    means = []
    for _ in range(n_resamples):
        sample = np.random.choice(values, size=len(values), replace=True)
        means.append(np.mean(sample))
    means = sorted(means)
    alpha = (1 - ci) / 2
    lower = means[int(alpha * n_resamples)]
    upper = means[int((1 - alpha) * n_resamples)]
    return {
        "mean": round(float(np.mean(values)), 4),
        "std": round(float(np.std(values, ddof=1)), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "n_folds": len(values),
        "per_fold": [round(float(v), 4) for v in values],
    }

def run_bootstrap():
    with open(DATA / "model_results.json") as f:
        data = json.load(f)

    results = {}
    for oname, odata in data.items():
        outcome_results = {}
        for key, res in odata.items():
            mc = res.get("metrics_cv", {})
            if not mc:
                continue
            model_result = {}
            for metric in ["accuracy", "f1", "roc_auc"]:
                folds_key = f"{metric}_folds"
                if folds_key in mc:
                    model_result[metric] = bootstrap_ci(mc[folds_key])
            if model_result:
                outcome_results[key] = model_result
        if outcome_results:
            results[oname] = outcome_results

    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 60)
    print("BOOTSTRAP CONFIDENCE INTERVALS (5-fold CV)")
    print("=" * 60)
    for oname, odata in results.items():
        print(f"\n--- {oname} ---")
        for key, metrics in odata.items():
            print(f"  {key}:")
            for metric, ci_data in metrics.items():
                print(f"    {metric}: {ci_data['mean']:.4f} [{ci_data['ci_lower']:.4f}, {ci_data['ci_upper']:.4f}] "
                      f"(std={ci_data['std']:.4f}, {ci_data['n_folds']} folds)")

    print(f"\nOutput saved to {OUT}")
    return results

if __name__ == "__main__":
    run_bootstrap()
