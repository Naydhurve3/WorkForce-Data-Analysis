import itertools
import warnings

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import chi2_contingency, f_oneway, pearsonr, spearmanr
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=RuntimeWarning)


def _is_binary(s):
    return s.nunique() == 2


def _is_numeric(s):
    return pd.api.types.is_numeric_dtype(s)


def _select_test(f1_type, f2_type, outcome_type):
    if outcome_type == "binary":
        if f1_type == "cat" or f2_type == "cat":
            return "chi2"
        return "logistic"
    if f1_type == "num" and f2_type == "num":
        return "pearson"
    if f1_type == "num" or f2_type == "num":
        return "anova"
    return "chi2"


def _chi2_test(f1, f2, outcome):
    p_vals = []
    stats = []
    for o in outcome.unique():
        mask = outcome == o
        ct = pd.crosstab(f1[mask], f2[mask])
        if ct.size == 0 or ct.shape != (1, 1):
            try:
                s, p, _, _ = chi2_contingency(ct)
                p_vals.append(p)
                stats.append(s)
            except Exception:
                pass
    if not p_vals:
        return 1.0, 0.0, 0.0
    return float(np.min(p_vals)), float(np.mean(stats)), 0.0


def _anova_test(num, cat):
    groups = [num[cat == v].dropna().values for v in cat.unique() if (cat == v).sum() > 1]
    if len(groups) < 2:
        return 1.0, 0.0, 0.0
    try:
        f_stat, p_val = f_oneway(*groups)
        if np.isnan(p_val):
            return 1.0, 0.0, 0.0
        ss_between = sum(len(g) * (np.mean(g) - np.mean(num.dropna())) ** 2 for g in groups)
        ss_total = sum((g - np.mean(num.dropna())) ** 2 for g in groups)
        eta_sq = ss_between / ss_total if ss_total > 0 else 0.0
        return float(p_val), float(f_stat), float(eta_sq)
    except Exception:
        return 1.0, 0.0, 0.0


def _pearson_test(a, b):
    mask = a.notna() & b.notna()
    if mask.sum() < 10:
        return 1.0, 0.0, 0.0
    try:
        r, p = pearsonr(a[mask], b[mask])
        if np.isnan(p):
            return 1.0, 0.0, 0.0
        return float(p), float(r), float(r ** 2)
    except Exception:
        return 1.0, 0.0, 0.0


def _encode_if_needed(s):
    if pd.api.types.is_numeric_dtype(s):
        return s
    le = LabelEncoder()
    return pd.Series(le.fit_transform(s.astype(str)), index=s.index)


def _mutual_info(features, outcome):
    clean = features.dropna()
    if len(clean) < 10:
        return 0.0
    outcome_clean = outcome.loc[clean.index].dropna()
    if len(outcome_clean) < 10:
        return 0.0
    common = clean.index.intersection(outcome_clean.index)
    if len(common) < 10:
        return 0.0
    x = _encode_if_needed(clean.loc[common])
    y = _encode_if_needed(outcome.loc[common])
    try:
        if _is_binary(outcome):
            mi = mutual_info_classif(x.values.reshape(-1, 1), y.values, random_state=42)
        else:
            mi = mutual_info_regression(x.values.reshape(-1, 1), y.values, random_state=42)
        return float(mi[0])
    except Exception:
        return 0.0


def _compute_impact(p_value, effect_size):
    if p_value <= 0 or np.isnan(p_value):
        return 0.0
    return round(-np.log10(max(p_value, 1e-300)) * max(effect_size, 0.001), 4)


class InteractionMiner:
    def __init__(self, config, random_state=42):
        self.cfg = config
        self.random_state = random_state

    def derive_outcomes(self, raw_df, feature_df):
        outcomes = pd.DataFrame(index=raw_df.index)
        outcomes["is_terminated"] = (
            raw_df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
        )
        outcomes["PerfScore"] = feature_df["PerfScore"]

        pay_map = {"Zone A": 1, "Zone B": 2, "Zone C": 3}
        outcomes["PayZone_encoded"] = raw_df["PayZone"].map(pay_map).fillna(0).astype(int)

        dept_gender = raw_df.groupby("DepartmentType")["GenderCode"]
        for dept, group in dept_gender:
            majority = group.value_counts().index[0]
            mask = raw_df["DepartmentType"] == dept
            outcomes.loc[mask, "is_minority_dept"] = (
                raw_df.loc[mask, "GenderCode"] != majority
            ).astype(int)
        outcomes["is_minority_dept"] = outcomes["is_minority_dept"].fillna(0).astype(int)

        outcomes["SeniorityLevel"] = feature_df["SeniorityLevel"]
        return outcomes

    def _classify(self, s):
        if _is_numeric(s) and s.nunique() > 5:
            return "num"
        return "cat"

    def test_pair(self, f1_vals, f2_vals, outcome_vals):
        clean = pd.DataFrame({"f1": f1_vals, "f2": f2_vals, "o": outcome_vals}).dropna()
        if len(clean) < 30:
            return None

        t1, t2, to = self._classify(clean["f1"]), self._classify(clean["f2"]), self._classify(clean["o"])
        method = _select_test(t1, t2, "binary" if clean["o"].nunique() == 2 else "numeric")

        if method == "anova":
            p_val, stat, eff = _anova_test(clean["f1"] if t1 == "num" else clean["f2"],
                                            clean["f2"] if t2 == "cat" else clean["f1"])
        elif method == "pearson":
            p_val, stat, eff = _pearson_test(clean["f1"], clean["f2"])
        elif method == "chi2":
            p_val, stat, eff = _chi2_test(clean["f1"], clean["f2"], clean["o"])
        else:
            p_val, stat, eff = 1.0, 0.0, 0.0

        mi = _mutual_info(clean["f1"], clean["f2"])
        impact = _compute_impact(p_val, eff if eff > 0 else mi)

        return {
            "method": method, "p_value": p_val, "statistic": round(float(stat), 4),
            "effect_size": round(float(eff), 4), "mutual_info": round(float(mi), 4),
            "impact": impact, "n": len(clean),
            "f1_type": t1, "f2_type": t2, "outcome_type": to,
        }

    def test_all_pairs(self, feature_df, outcomes):
        feature_names = feature_df.columns.tolist()
        outcome_names = outcomes.columns.tolist()
        pairs = list(itertools.combinations(feature_names, 2))
        total = len(pairs) * len(outcome_names)
        logger.info(f"Testing {len(pairs)} feature pairs × {len(outcome_names)} outcomes = {total} tests")

        results = []
        count = 0
        for f1, f2 in pairs:
            for o_name in outcome_names:
                res = self.test_pair(feature_df[f1], feature_df[f2], outcomes[o_name])
                if res is not None:
                    res["feature_1"] = f1
                    res["feature_2"] = f2
                    res["outcome"] = o_name
                    results.append(res)
                count += 1
                if count % 500 == 0:
                    logger.info(f"  Progress: {count}/{total} tests completed")

        if not results:
            logger.warning("No valid interaction tests completed")
            return pd.DataFrame()

        rdf = pd.DataFrame(results)
        rdf["significant"] = rdf["p_value"] < 0.05
        rdf["bonferroni"] = rdf["p_value"] < (0.05 / max(len(rdf), 1))
        rdf = rdf.sort_values("impact", ascending=False).reset_index(drop=True)
        rdf["rank"] = range(1, len(rdf) + 1)
        logger.info(f"  Found {rdf['significant'].sum()} significant at p<0.05, {rdf['bonferroni'].sum()} at Bonferroni level")
        return rdf

    def run_segmentation(self, feature_df, outcome_name, f1, f2, max_depth=3):
        clean = feature_df[[f1, f2]].dropna()
        if len(clean) < 30:
            return None
        o = self.outcomes.loc[clean.index, outcome_name].dropna()
        common = clean.index.intersection(o.index)
        if len(common) < 30:
            return None

        x = pd.DataFrame()
        for c in [f1, f2]:
            if _is_numeric(clean.loc[common, c]):
                x[c] = clean.loc[common, c]
            else:
                x[c] = LabelEncoder().fit_transform(clean.loc[common, c].astype(str))

        y = _encode_if_needed(o.loc[common])

        if _is_binary(pd.Series(y)):
            clf = DecisionTreeClassifier(max_depth=max_depth, random_state=self.random_state, min_samples_leaf=10)
        else:
            clf = DecisionTreeRegressor(max_depth=max_depth, random_state=self.random_state, min_samples_leaf=10)
        clf.fit(x.values, y.values)

        n_leaves = clf.get_n_leaves()
        importances = dict(zip([f1, f2], clf.feature_importances_))
        return {"tree": clf, "n_leaves": n_leaves, "importances": importances, "feature_names": [f1, f2]}

    def run(self, raw_df, feature_df):
        logger.info("=" * 50)
        logger.info("Phase 3: Interaction Mining")
        logger.info("=" * 50)

        self.outcomes = self.derive_outcomes(raw_df, feature_df)
        logger.info(f"Derived {len(self.outcomes.columns)} outcomes: {list(self.outcomes.columns)}")

        results = self.test_all_pairs(feature_df, self.outcomes)

        self.results = results
        return results
