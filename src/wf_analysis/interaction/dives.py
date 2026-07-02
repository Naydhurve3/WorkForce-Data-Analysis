import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import ttest_ind


def _find_dominant_segment(feature_df, f1, f2, outcome_series, n_groups=3):
    combo = feature_df[f1].astype(str) + "___" + feature_df[f2].astype(str)
    stats = outcome_series.groupby(combo).agg(["mean", "count", "std"])
    stats = stats[stats["count"] >= 15].sort_values("mean", ascending=False)
    if stats.empty:
        return None, None, None
    best = stats.index[0]
    mask = combo == best
    f1_val, f2_val = best.split("___", 1)
    return mask, best, (f1_val, f2_val)


def analyze_dive(feature_df, raw_df, f1, f2, outcome_name, outcome_series, top_n_groups=3):
    logger.info(f"Dive: {f1} x {f2} -> {outcome_name}")
    mask, label, (f1_val, f2_val) = _find_dominant_segment(feature_df, f1, f2, outcome_series, top_n_groups)
    if mask is None:
        logger.warning("  No dominant segment found")
        return None

    n_seg, n_rest = mask.sum(), (~mask).sum()
    seg_mean = float(outcome_series[mask].mean())
    rest_mean = float(outcome_series[~mask].mean())
    diff = seg_mean - rest_mean

    try:
        t_stat, p_val = ttest_ind(outcome_series[mask], outcome_series[~mask], equal_var=False)
        p_val = float(p_val)
    except Exception:
        t_stat, p_val = 0.0, 1.0

    effect = diff / float(outcome_series[~mask].std()) if outcome_series[~mask].std() > 0 else 0

    segment_df = feature_df.loc[mask]
    rest_df = feature_df.loc[~mask]
    profile = {}
    for c in segment_df.select_dtypes(include=[np.number]).columns:
        profile[c] = {
            "segment_mean": float(segment_df[c].mean()),
            "rest_mean": float(rest_df[c].mean()),
            "segment_std": float(segment_df[c].std()),
        }

    whatif = []
    vals = feature_df[f1].astype(str).unique()[:5]
    for v in vals:
        sub = feature_df[f1].astype(str) == v
        w_mean = float(outcome_series[sub].mean()) if sub.sum() > 10 else None
        w_count = int(sub.sum())
        whatif.append({"f1_val": v, "outcome_mean": w_mean, "count": w_count})

    result = {
        "f1": f1, "f2": f2, "f1_val": f1_val, "f2_val": f2_val,
        "outcome": outcome_name, "segment_label": label,
        "n_segment": int(n_seg), "n_rest": int(n_rest),
        "segment_outcome_mean": round(seg_mean, 3),
        "rest_outcome_mean": round(rest_mean, 3),
        "difference": round(diff, 3),
        "effect_size": round(effect, 4),
        "p_value": p_val,
        "profile": profile,
        "whatif": whatif,
        "segment_mask": mask,
    }
    logger.info(f"  Segment: {label} (n={n_seg}) | outcome: seg={seg_mean:.3f} vs rest={rest_mean:.3f} | d={effect:.3f} p={p_val:.4f}")
    return result


def find_dives(feature_df, raw_df, outcome_defs, top_results, n_dives=5):
    dives = []
    used = set()
    outcome_df = pd.DataFrame(outcome_defs)

    for _, row in top_results.iterrows():
        if len(dives) >= n_dives:
            break
        key = (row["feature_1"], row["feature_2"], row["outcome"])
        if key in used:
            continue
        used.add(key)
        o_series = outcome_df[row["outcome"]]
        dive = analyze_dive(feature_df, raw_df, row["feature_1"], row["feature_2"], row["outcome"], o_series)
        if dive is not None:
            dive["impact"] = row["impact"]
            dives.append(dive)

    logger.info(f"Total deep dives: {len(dives)}")
    return dives
