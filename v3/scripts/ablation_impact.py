"""Ablation analysis: compare impact score variants for rank stability."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path
import json

DATA = Path("data") / "interaction"
OUT = DATA / "ablation_ranking.json"

def compute_variants():
    ir = pd.read_parquet(DATA / "interaction_results.parquet").copy()

    # Score variants
    ir["S_impact"] = ir["impact"].fillna(0)
    ir["S_mi"] = ir["mutual_info"].fillna(0)
    ir["S_stat"] = ir["statistic"].fillna(0).abs()
    ir["S_mi_log"] = ir["mutual_info"].fillna(0) * np.log(ir["n"].fillna(1))

    # Normalized variants
    for col in ["S_impact", "S_mi", "S_stat", "S_mi_log"]:
        mx = ir[col].max()
        if mx > 0:
            ir[f"{col}_norm"] = ir[col] / mx
        else:
            ir[f"{col}_norm"] = ir[col]

    ir["S_additive"] = ir["S_mi_norm"] + ir["S_stat_norm"] + np.log(ir["n"].fillna(1)).apply(lambda x: min(x, 10)) / 10
    ir["S_ensemble"] = ir["S_mi_norm"] * ir["S_stat_norm"] * np.log(ir["n"].fillna(1)).apply(lambda x: min(x, 10)) / 10

    score_cols = ["S_impact", "S_mi", "S_stat", "S_mi_log", "S_additive", "S_ensemble"]
    score_labels = ["Impact (full)", "MI only", "Statistic only", "MI x log(n)", "Additive (norm)", "Ensemble (norm)"]

    results = {}
    for outcome in ir["outcome"].unique():
        odf = ir[ir["outcome"] == outcome].copy()
        outcome_res = {"n_pairs": len(odf)}

        rankings = {}
        for col, label in zip(score_cols, score_labels):
            odf[f"rank_{label}"] = odf[col].rank(ascending=False, method="min")
            rankings[label] = odf.nsmallest(10, f"rank_{label}")[["feature_1", "feature_2", col]].to_dict("records")

        corr = {}
        for i, l1 in enumerate(score_labels):
            for l2 in score_labels[i+1:]:
                r_s, p_s = spearmanr(odf[f"rank_{l1}"], odf[f"rank_{l2}"])
                corr[f"{l1} vs {l2}"] = round(float(r_s), 4)

        outcome_res["correlations"] = corr
        outcome_res["top_10"] = rankings
        results[outcome] = outcome_res

    agg = {"overall": {"n_outcomes": len(results)}}
    all_corrs = []
    for oname, ores in results.items():
        for pair, rho in ores["correlations"].items():
            all_corrs.append(rho)
    agg["overall"]["mean_rho"] = round(float(np.mean(all_corrs)), 4)
    agg["overall"]["min_rho"] = round(float(np.min(all_corrs)), 4)

    output = {"per_outcome": results, "aggregate": agg, "score_labels": score_labels}
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print("=" * 60)
    print("IMPACT SCORE ABLATION")
    print("=" * 60)
    for oname, ores in results.items():
        print(f"\n--- {oname} ({ores['n_pairs']} pairs) ---")
        for pair, rho in ores["correlations"].items():
            print(f"  {pair:35s}: Spearman rho={rho:.4f}")

    print(f"\nAggregate: mean Spearman={agg['overall']['mean_rho']:.4f}")
    print(f"Output saved to {OUT}")
    return results

if __name__ == "__main__":
    compute_variants()
