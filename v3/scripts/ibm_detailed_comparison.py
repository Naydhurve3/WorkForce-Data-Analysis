"""Cross-dataset comparison: synthetic vs IBM interaction rankings + model metrics."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import json

SYN_DATA = Path("data") / "interaction"
IBM_DATA = Path("data") / "benchmark_ibm"
OUT = IBM_DATA / "ibm_vs_synthetic_comparison.json"

def compare():
    syn_ir = pd.read_parquet(SYN_DATA / "interaction_results.parquet")
    with open(SYN_DATA / "model_results.json") as f:
        syn_models = json.load(f)
    ibm_ir = pd.read_csv(IBM_DATA / "ibm_interaction_rankings.csv")
    ibm_models = pd.read_csv(IBM_DATA / "ibm_model_metrics.csv")

    # Shared feature overlap (how many feature pairs appear in both datasets)
    syn_features = set(syn_ir["feature_1"].unique()) | set(syn_ir["feature_2"].unique())
    ibm_features = set(ibm_ir["feature_1"].unique()) | set(ibm_ir["feature_2"].unique())
    shared_features = syn_features & ibm_features

    # Compare model metrics (termination vs attrition)
    syn_termination = {}
    for oname, odata in syn_models.items():
        if "terminated" in oname:
            for key, res in odata.items():
                mc = res.get("metrics_cv", res.get("metrics", {}))
                sk = key.replace("_without", "").replace("_with", "")
                variant = "without" if key.endswith("_without") else "with"
                syn_termination[f"{sk}_{variant}"] = {
                    "accuracy_mean": mc.get("accuracy_mean", mc.get("accuracy")),
                    "accuracy_std": mc.get("accuracy_std", 0),
                    "f1_mean": mc.get("f1_mean", mc.get("f1")),
                    "f1_std": mc.get("f1_std", 0),
                    "roc_auc_mean": mc.get("roc_auc_mean", mc.get("roc_auc")),
                    "roc_auc_std": mc.get("roc_auc_std", 0),
                }

    ibm_termination = {}
    for _, r in ibm_models[ibm_models["variant"].isin(["without", "with"])].iterrows():
        key = f"{r['model']}_{r['variant']}"
        ibm_termination[key] = {
            "accuracy_mean": r["accuracy_mean"],
            "accuracy_std": r["accuracy_std"],
            "f1_mean": r["f1_mean"],
            "f1_std": r["f1_std"],
            "roc_auc_mean": r.get("roc_auc_mean"),
            "roc_auc_std": r.get("roc_auc_std"),
        }

    # Cross-dataset rank correlation (for shared statistical methods)
    syn_methods = syn_ir.groupby(["feature_1", "feature_2", "method"]).agg({"statistic": "first"}).reset_index()
    ibm_methods = ibm_ir.groupby(["feature_1", "feature_2", "method"]).agg({"statistic": "first"}).reset_index()

    syn_lookup = {}
    for _, r in syn_methods.iterrows():
        syn_lookup[(r["feature_1"], r["feature_2"], r["method"])] = r["statistic"]

    ibm_lookup = {}
    for _, r in ibm_methods.iterrows():
        ibm_lookup[(r["feature_1"], r["feature_2"], r["method"])] = r["statistic"]

    shared_pairs = set(syn_lookup.keys()) & set(ibm_lookup.keys())
    if shared_pairs:
        syn_vals = [syn_lookup[p] for p in shared_pairs]
        ibm_vals = [ibm_lookup[p] for p in shared_pairs]
        rho, p_val = spearmanr(syn_vals, ibm_vals)
        rank_corr = {
            "n_shared_pairs": len(shared_pairs),
            "spearman_rho": round(float(rho), 4),
            "spearman_p": float(p_val),
        }
    else:
        rank_corr = {"n_shared_pairs": 0, "note": "No shared feature pairs found"}

    output = {
        "synthetic": {
            "n_features": len(syn_features),
            "n_pairs": len(syn_ir),
        },
        "ibm": {
            "n_features": len(ibm_features),
            "n_pairs": len(ibm_ir),
        },
        "shared_features": len(shared_features),
        "shared_feature_list": sorted(list(shared_features)),
        "rank_correlation": rank_corr,
        "model_comparison": {
            "synthetic_termination": syn_termination,
            "ibm_termination": ibm_termination,
        },
    }

    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("=" * 60)
    print("CROSS-DATASET COMPARISON: Synthetic vs IBM HR")
    print("=" * 60)
    print(f"Synthetic features: {len(syn_features)}, IBM features: {len(ibm_features)}")
    print(f"Shared features: {len(shared_features)}")
    if rank_corr.get("n_shared_pairs", 0) > 0:
        print(f"Rank correlation (shared pairs): Spearman={rank_corr['spearman_rho']:.4f} "
              f"(p={rank_corr['spearman_p']:.4f}, n={rank_corr['n_shared_pairs']})")

    print("\nModel metrics comparison (termination/attrition):")
    print(f"{'Model':20s} {'Synthetic acc':15s} {'IBM acc':15s}")
    for key in sorted(set(list(syn_termination.keys()) + list(ibm_termination.keys()))):
        s = syn_termination.get(key, {})
        i = ibm_termination.get(key, {})
        s_acc = f"{s.get('accuracy_mean', 'N/A'):.4f}" if s.get('accuracy_mean') else "N/A"
        i_acc = f"{i.get('accuracy_mean', 'N/A'):.4f}" if i.get('accuracy_mean') else "N/A"
        print(f"  {key:20s} {s_acc:>15s} {i_acc:>15s}")

    print(f"\nOutput saved to {OUT}")
    return output

if __name__ == "__main__":
    compare()
