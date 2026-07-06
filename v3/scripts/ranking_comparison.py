"""Compare Impact Score rankings against individual component rankings."""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, kendalltau
from pathlib import Path

DATA = Path("data") / "interaction"
OUT = DATA / "ranking_comparison.json"

def _jaccard(a, b):
    a_set, b_set = set(a), set(b)
    return len(a_set & b_set) / len(a_set | b_set) if (a_set | b_set) else 0

def compare_rankings():
    ir = pd.read_parquet(DATA / "interaction_results.parquet")

    ranking_methods = {
        "Impact Score": "impact",
        "Mutual Info": "mutual_info",
        "Statistic": "statistic",
        "Sample Size": "n",
    }

    results = {}
    for outcome in ir["outcome"].unique():
        odf = ir[ir["outcome"] == outcome].copy()
        outcome_results = {}

        for label, col in ranking_methods.items():
            odf[f"rank_{label}"] = odf[col].rank(ascending=False, method="min")

        outcome_results["n_pairs"] = len(odf)
        outcome_results["n_significant"] = int(odf["significant"].sum())
        outcome_results["n_bonferroni"] = int(odf["bonferroni"].sum())

        corr = {}
        labels = list(ranking_methods.keys())
        for i, l1 in enumerate(labels):
            for l2 in labels[i+1:]:
                pair_key = f"{l1} vs {l2}"
                r_s, p_s = spearmanr(odf[f"rank_{l1}"], odf[f"rank_{l2}"])
                r_k, p_k = kendalltau(odf[f"rank_{l1}"], odf[f"rank_{l2}"])
                corr[pair_key] = {
                    "spearman_rho": round(float(r_s), 4),
                    "spearman_p": float(p_s),
                    "kendall_tau": round(float(r_k), 4),
                    "kendall_p": float(p_k),
                }

        top_k = {}
        for k in [10, 20, 50]:
            overlap = {}
            for i, l1 in enumerate(labels):
                top1 = set(odf.nsmallest(k, f"rank_{l1}")[["feature_1", "feature_2"]].itertuples(index=False, name=None))
                for l2 in labels[i+1:]:
                    top2 = set(odf.nsmallest(k, f"rank_{l2}")[["feature_1", "feature_2"]].itertuples(index=False, name=None))
                    j = _jaccard(top1, top2)
                    overlap[f"{l1} vs {l2}"] = round(j, 4)
            top_k[f"top_{k}_jaccard"] = overlap

        outcome_results["correlations"] = corr
        outcome_results["top_k_overlap"] = top_k
        results[outcome] = outcome_results

    # Aggregate across outcomes
    agg = {"overall": {"n_outcomes": len(results)}}
    all_corrs = []
    for oname, ores in results.items():
        for pair, cvals in ores["correlations"].items():
            all_corrs.append(cvals["spearman_rho"])
    agg["overall"]["mean_spearman"] = round(float(np.mean(all_corrs)), 4)
    agg["overall"]["min_spearman"] = round(float(np.min(all_corrs)), 4)
    agg["overall"]["max_spearman"] = round(float(np.max(all_corrs)), 4)

    import json
    output = {"per_outcome": results, "aggregate": agg}
    with open(OUT, "w") as f:
        json.dump(output, f, indent=2)

    print("=" * 60)
    print("RANKING METHOD COMPARISON")
    print("=" * 60)
    for oname, ores in results.items():
        print(f"\n--- {oname} ({ores['n_pairs']} pairs, {ores['n_significant']} significant) ---")
        for pair, cvals in ores["correlations"].items():
            print(f"  {pair:30s}: Spearman={cvals['spearman_rho']:.4f} Kendall={cvals['kendall_tau']:.4f}")
        for k, overlaps in ores["top_k_overlap"].items():
            print(f"  {k}:")
            for pair, j in overlaps.items():
                print(f"    {pair:30s}: Jaccard={j:.4f}")

    cov_col = [c for c in ir.columns if "impact" in c or "mutual" in c or "stat" in c or c == "n"]
    print(f"\nAggregate: mean Spearman={agg['overall']['mean_spearman']:.4f} across {agg['overall']['n_outcomes']} outcomes")
    print(f"Output saved to {OUT}")
    return results

if __name__ == "__main__":
    compare_rankings()
