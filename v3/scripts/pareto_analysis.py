"""
Pareto / Ablation Analysis: What fraction of interactions drive the impact?
Also computes model accuracy comparison WITH vs WITHOUT interactions.
"""

import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path("data") / "interaction"


def pareto_interactions():
    """Top-N cumulative impact analysis from parquet."""
    df = pd.read_parquet(DATA_DIR / "interaction_results.parquet")
    df = df.sort_values("impact", ascending=False).reset_index(drop=True)
    total = df["impact"].sum()
    df["cumulative_pct"] = df["impact"].cumsum() / total * 100
    df["rank"] = range(1, len(df) + 1)

    print("=" * 60)
    print("PARETO ANALYSIS - Interaction Impact Concentration")
    print("=" * 60)
    print(f"Total pairs: {len(df)}")
    print(f"Total impact: {total:.2f}")
    print()
    for pct in [10, 25, 50, 80, 90, 95, 99]:
        idx = (df["cumulative_pct"] >= pct).idxmax()
        n = df.loc[idx, "rank"]
        print(f"  Top {int(n):4d} ({n/len(df)*100:.1f}%) -> {pct}% of impact")

    print("\n--- Per-outcome ---")
    for outcome in df["outcome"].unique():
        odf = df[df["outcome"] == outcome].reset_index(drop=True)
        ot = odf["impact"].sum()
        odf["cp"] = odf["impact"].cumsum() / ot * 100
        n80 = (odf["cp"] >= 80).idxmax() + 1
        print(f"  {outcome:25s}: {len(odf):3d} pairs, top {n80:2d} -> 80% impact")

    # Top-10 share of total
    top10 = df.head(10)["impact"].sum()
    print(f"\n  Top 10 pairs: {top10/total*100:.1f}% of total impact")
    top50 = df.head(50)["impact"].sum()
    print(f"  Top 50 pairs: {top50/total*100:.1f}% of total impact")

    df.to_csv(DATA_DIR / "pareto_analysis.csv", index=False)
    return df


def model_comparison():
    """Extract model metrics from model_results.json (CV format)."""
    with open(DATA_DIR / "model_results.json") as f:
        data = json.load(f)

    print("\n" + "=" * 60)
    print("MODEL ACCURACY COMPARISON (5-fold CV)")
    print("=" * 60)
    rows = []
    for oname, odata in data.items():
        for key, res in odata.items():
            mc = res.get("metrics_cv", res.get("metrics", {}))
            variant = "WITH" if key.endswith("_with") else "WITHOUT"
            sk = key.replace("_without", "").replace("_with", "")

            acc_mean = mc.get("accuracy_mean", mc.get("accuracy", 0))
            acc_std = mc.get("accuracy_std", 0)
            f1_mean = mc.get("f1_mean", mc.get("f1", 0))
            f1_std = mc.get("f1_std", 0)
            auc_mean = mc.get("roc_auc_mean", mc.get("roc_auc", ""))
            auc_std = mc.get("roc_auc_std", "")

            rows.append({
                "outcome": oname,
                "model": sk,
                "variant": variant,
                "accuracy_mean": acc_mean,
                "accuracy_std": acc_std,
                "f1_mean": f1_mean,
                "f1_std": f1_std,
                "roc_auc_mean": auc_mean,
                "roc_auc_std": auc_std,
            })
    pdf = pd.DataFrame(rows)
    for _, r in pdf.iterrows():
        auc_str = f" auc={r['roc_auc_mean']:.4f}" if r['roc_auc_mean'] != "" else ""
        std_s = f"\u00b1{r['accuracy_std']:.4f}" if r['accuracy_std'] > 0 else ""
        print(f"  {r['outcome']:22s} {r['model']:4s} ({r['variant']:7s}): acc={r['accuracy_mean']:.4f}{std_s} f1={r['f1_mean']:.4f}{auc_str}")

    # Summary deltas using accuracy_mean
    print("\n--- Accuracy Delta (WITH - WITHOUT) ---")
    comp = pdf.groupby(["outcome", "variant"])["accuracy_mean"].mean().unstack()
    if "WITH" in comp and "WITHOUT" in comp:
        comp["delta"] = (comp["WITH"] - comp["WITHOUT"]) * 100
        for outcome, row in comp.iterrows():
            print(f"  {outcome:22s}: Delta = {row['delta']:+.2f} pp")

    pdf.to_csv(DATA_DIR / "model_metrics_comparison.csv", index=False)
    return pdf


def ibm_comparison():
    """Compare IBM benchmark results against synthetic."""
    ibm_path = Path("data") / "benchmark_ibm" / "ibm_summary.json"
    if not ibm_path.exists():
        print("\n[SKIP] IBM benchmark not found")
        return
    with open(ibm_path) as f:
        ibm = json.load(f)

    # Compute synthetic best from CV results
    syn_best_without = 0
    syn_best_with = 0
    try:
        with open(DATA_DIR / "model_results.json") as f:
            syn_data = json.load(f)
        for oname, odata in syn_data.items():
            for key, res in odata.items():
                mc = res.get("metrics_cv", {})
                acc = mc.get("accuracy_mean", 0)
                if "_without" in key:
                    syn_best_without = max(syn_best_without, acc)
                if "_with" in key:
                    syn_best_with = max(syn_best_with, acc)
    except Exception:
        pass

    # Synthetic top impact from parquet
    syn_top_impact = "N/A"
    syn_top_pair = "N/A"
    try:
        ir = pd.read_parquet(DATA_DIR / "interaction_results.parquet")
        t = ir.sort_values("impact", ascending=False).iloc[0]
        syn_top_impact = f"{t['impact']:.1f}"
        syn_top_pair = f"{t['feature_1'][:10]}x{t['feature_2'][:10]}"
    except Exception:
        pass

    print("\n" + "=" * 60)
    print("BENCHMARK COMPARISON: Synthetic vs IBM HR")
    print("=" * 60)
    print(f"{'Metric':35s} {'Synthetic':>12s} {'IBM HR':>12s}")
    print("-" * 61)
    print(f"{'Rows':35s} {'3,000':>12s} {str(ibm['n_rows']):>12s}")
    print(f"{'Best acc (WITHOUT int.)':35s} {syn_best_without:>12.4f} {str(ibm['best_acc_without']):>12s}")
    print(f"{'Best acc (WITH int.)':35s} {syn_best_with:>12.4f} {str(ibm['best_acc_with']):>12s}")
    print(f"{'Top interaction impact':35s} {syn_top_impact:>12s} {str(ibm['top_interaction']['impact']):>12s}")
    print(f"{'Top feature pair':35s} {syn_top_pair:>12s} {ibm['top_interaction']['feature_1'][:10]+'x':>12s}")


def main():
    df = pareto_interactions()
    pdf = model_comparison()
    ibm_comparison()
    print("\nDone - saved to data/interaction/pareto_analysis.csv")


if __name__ == "__main__":
    main()
