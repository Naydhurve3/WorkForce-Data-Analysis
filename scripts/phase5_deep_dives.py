"""Phase 5: Deep Dives — top-5 discoveries with subgroup, profile, what-if analysis."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.figures import EDAFigureFactory
from wf_analysis.interaction.dives import find_dives


def main():
    logger.info("=" * 60)
    logger.info("  Phase 5: Deep Dives — Subgroup / Profile / What-If")
    logger.info("=" * 60)

    cfg = InteractionConfig()

    raw_df = DataLoader.load(cfg.raw_path, validate=False)
    engineer = FeatureEngineer(cfg)
    feature_df = engineer.compute_all(raw_df)

    interaction_results = None
    ir_path = Path("data/interaction/interaction_results.parquet")
    if ir_path.exists():
        interaction_results = pd.read_parquet(ir_path)

    outcome_defs = {
        "is_terminated": raw_df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int),
        "PerfScore": feature_df.get("PerfScore", raw_df["Current Employee Rating"].fillna(0).astype(int)),
        "PayZone_encoded": raw_df["PayZone"].map({"Zone A": 1, "Zone B": 2, "Zone C": 3}).fillna(0).astype(int),
    }
    try:
        dept_gender = raw_df.groupby("DepartmentType")["GenderCode"]
        for dept, group in dept_gender:
            majority = group.value_counts().index[0]
            mask = raw_df["DepartmentType"] == dept
            outcome_defs["is_minority_dept"] = outcome_defs.get("is_minority_dept", pd.Series(0, index=raw_df.index))
            outcome_defs["is_minority_dept"].loc[mask] = (raw_df.loc[mask, "GenderCode"] != majority).astype(int)
        outcome_defs["is_minority_dept"] = outcome_defs["is_minority_dept"].fillna(0).astype(int)
    except Exception:
        outcome_defs["is_minority_dept"] = pd.Series(0, index=raw_df.index)
    outcome_defs["SeniorityLevel"] = feature_df.get("SeniorityLevel", pd.Series(1, index=raw_df.index))

    if interaction_results is not None:
        top_per_outcome = (
            interaction_results.loc[interaction_results.groupby("outcome")["impact"].idxmax()]
            .sort_values("impact", ascending=False)
        )
        top_per_outcome = top_per_outcome.drop_duplicates(subset=["feature_1", "feature_2", "outcome"])
        logger.info(f"\nTop discoveries (one per outcome):\n{top_per_outcome[['feature_1','feature_2','outcome','impact']].to_string(index=False)}")
        dives = find_dives(feature_df, raw_df, outcome_defs, top_per_outcome, n_dives=5)
    else:
        fallback = pd.DataFrame([
            {"feature_1": "JobFamily", "feature_2": "DepartmentType", "outcome": "PayZone_encoded", "impact": 181.3},
            {"feature_1": "TenureYears", "feature_2": "ExitQuarter", "outcome": "is_minority_dept", "impact": 107.7},
            {"feature_1": "Region", "feature_2": "DepartmentType", "outcome": "PerfScore", "impact": 62.8},
            {"feature_1": "Region", "feature_2": "IntersectionalID", "outcome": "is_terminated", "impact": 55.8},
            {"feature_1": "JobFamily", "feature_2": "IsManager", "outcome": "SeniorityLevel", "impact": 68.2},
        ])
        dives = find_dives(feature_df, raw_df, outcome_defs, fallback, n_dives=5)

    if not dives:
        logger.error("No dives could be computed — aborting Phase 5")
        return

    ff = EDAFigureFactory(cfg)

    for i, dive in enumerate(dives):
        base_num = 37 + i * 3
        logger.info(f"Figure {base_num:02d}: Subgroup — {dive['f1']}×{dive['f2']}→{dive['outcome']}")
        ff.plot_deep_dive_subgroup(dive, base_num)

        logger.info(f"Figure {base_num+1:02d}: Profile — {dive['f1_val']}×{dive['f2_val']}")
        ff.plot_deep_dive_profile(dive, feature_df, base_num + 1)

        logger.info(f"Figure {base_num+2:02d}: What-If — {dive['f1']}={dive['f1_val']}")
        ff.plot_deep_dive_whatif(dive, base_num + 2)

    logger.info("Figure 52: Deep Dive Dashboard")
    ff.figure_52_deep_dive_dashboard(dives)

    results_path = Path(cfg.output_dir) / "deep_dives.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for d in dives:
        entry = {k: v for k, v in d.items() if k != "segment_mask"}
        entry["profile"] = {k: v for k, v in entry.get("profile", {}).items()}
        entry["whatif"] = [
            {kk: vv for kk, vv in w.items()}
            for w in entry.get("whatif", [])
        ]
        serializable.append(entry)

    import json
    with open(results_path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"\nDeep dive results saved to {results_path}")

    logger.info("=" * 60)
    logger.info("  Phase 5 complete — 16 figures (37–52)")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
