import warnings
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import StackingClassifier
from lifelines import KaplanMeierFitter, CoxPHFitter
warnings.filterwarnings("ignore")


def _encode_interaction_pair(feature_df, f1, f2, top_n=20):
    cat_col = f1 + "__" + f2
    f1_num = pd.api.types.is_numeric_dtype(feature_df[f1]) and feature_df[f1].nunique() > 5
    f2_num = pd.api.types.is_numeric_dtype(feature_df[f2]) and feature_df[f2].nunique() > 5
    if f1_num and f2_num:
        feature_df[cat_col] = (feature_df[f1] * feature_df[f2]).fillna(0)
    else:
        feature_df[cat_col] = feature_df[f1].astype(str) + "__" + feature_df[f2].astype(str)
        le = LabelEncoder()
        feature_df[cat_col] = le.fit_transform(feature_df[cat_col].values)
    return feature_df


def _prepare_data(feature_df, outcome_series, interaction_pairs=None, n_interactions=20):
    x = feature_df.select_dtypes(include=[np.number, "int32", "int64", "float64"]).copy()
    x = x.fillna(x.median() if len(x) > 0 else 0)
    y = outcome_series
    if y.dtype == object or y.dtype.name == "category":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y.astype(str)), index=y.index, name=outcome_series.name)
    y = y.fillna(y.mode().iloc[0] if len(y) > 0 else 0).astype(int)
    common = x.index.intersection(y.index)
    x, y = x.loc[common], y.loc[common]
    if interaction_pairs and len(interaction_pairs) > 0:
        x_int = x.copy()
        for f1, f2 in interaction_pairs[:n_interactions]:
            if f1 in x_int.columns and f2 in x_int.columns:
                x_int = _encode_interaction_pair(x_int, f1, f2)
        return x, x_int, y
    return x, x, y


def _eval_model(y_true, y_pred, y_proba=None):
    result = {"accuracy": round(float(accuracy_score(y_true, y_pred)), 4)}
    n_classes = len(np.unique(y_true))
    if n_classes == 2:
        result["f1"] = round(float(f1_score(y_true, y_pred)), 4)
        if y_proba is not None:
            try:
                result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba[:, 1])), 4)
            except Exception:
                result["roc_auc"] = None
        else:
            result["roc_auc"] = None
    else:
        result["f1"] = round(float(f1_score(y_true, y_pred, average="weighted")), 4)
        result["roc_auc"] = None
    result["cm"] = confusion_matrix(y_true, y_pred).tolist()
    return result


class InteractionModeler:
    def __init__(self, config, random_state=42):
        self.cfg = config
        self.random_state = random_state
        self.models_w = {}
        self.models_wo = {}
        self.results = []

    def build_interaction_features(self, feature_df, top_interactions):
        logger.info(f"Building {len(top_interactions)} interaction features")
        for f1, f2 in top_interactions:
            if f1 in feature_df.columns and f2 in feature_df.columns:
                feature_df = _encode_interaction_pair(feature_df, f1, f2)
        return feature_df

    def train_and_compare(self, outcome_name, outcome_series, feature_df, interaction_pairs=None, n_interactions=15):
        logger.info(f"\n{'='*50}\nModeling: {outcome_name}\n{'='*50}")
        x_wo, x_w, y = _prepare_data(feature_df, outcome_series, interaction_pairs, n_interactions)
        has_interactions = x_w.shape[1] > x_wo.shape[1]

        if len(y.unique()) < 2:
            logger.warning(f"  Single class in target — skipping")
            return {}, {}

        x_train_wo, x_test_wo, y_train, y_test = train_test_split(x_wo, y, test_size=0.25, random_state=self.random_state, stratify=y)
        x_train_w, x_test_w = x_train_wo.copy(), x_test_wo.copy()
        if has_interactions:
            train_idx = x_train_wo.index
            test_idx = x_test_wo.index
            x_train_w = x_w.loc[train_idx]
            x_test_w = x_w.loc[test_idx]

        multi = len(y.unique()) > 2
        n_classes = len(y.unique())

        models = {
            "LR": LogisticRegression(max_iter=2000, random_state=self.random_state, multi_class="multinomial" if multi else "auto", class_weight="balanced"),
            "RF": RandomForestClassifier(n_estimators=200, random_state=self.random_state, n_jobs=-1, class_weight="balanced"),
        }
        if n_classes == 2:
            try:
                import xgboost as xgb
                models["XGB"] = xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, subsample=0.8,
                                                    colsample_bytree=0.8, random_state=self.random_state, eval_metric="logloss", verbosity=0)
            except ImportError:
                models["XGB"] = GradientBoostingClassifier(n_estimators=150, random_state=self.random_state, max_depth=4)

        outcome_results = {}
        for name, model in models.items():
            for variant, x_tr, x_te in [("without", x_train_wo, x_test_wo), ("with", x_train_w, x_test_w)]:
                if variant == "with" and not has_interactions:
                    continue
                try:
                    model.fit(x_tr, y_train)
                    y_pred = model.predict(x_te)
                    y_proba = model.predict_proba(x_te) if hasattr(model, "predict_proba") else None
                    metrics = _eval_model(y_test, y_pred, y_proba)
                    feat_imp = None
                    if hasattr(model, "feature_importances_"):
                        feat_imp = dict(zip(x_tr.columns, model.feature_importances_.round(4)))
                    elif hasattr(model, "coef_"):
                        if model.coef_.ndim > 1:
                            feat_imp = dict(zip(x_tr.columns, np.abs(model.coef_).mean(axis=0).round(4)))
                        else:
                            feat_imp = dict(zip(x_tr.columns, np.abs(model.coef_).round(4)))
                    coeff = model.coef_.tolist() if hasattr(model, "coef_") else None
                    key = f"{name}_{variant}"
                    outcome_results[key] = {
                        "model": model, "metrics": metrics, "feature_importance": feat_imp,
                        "y_pred": y_pred, "y_test": y_test, "y_proba": y_proba,
                        "x_test": x_te, "coefficients": coeff,
                        "feature_names": list(x_tr.columns),
                    }
                    logger.info(f"  {name} ({variant}): acc={metrics['accuracy']:.3f} f1={metrics.get('f1', 0):.3f}" +
                                (f" auc={metrics.get('roc_auc', 0):.3f}" if metrics.get("roc_auc") else ""))
                except Exception as e:
                    logger.warning(f"  {name} ({variant}) failed: {e}")

        logger.info(f"\n  WITHOUT interactions best: {max((v['metrics']['accuracy'] for k,v in outcome_results.items() if 'without' in k), default=0):.3f}")
        if has_interactions:
            logger.info(f"  WITH interactions best:    {max((v['metrics']['accuracy'] for k,v in outcome_results.items() if 'with' in k), default=0):.3f}")

        best_key = max(outcome_results, key=lambda k: outcome_results[k]["metrics"]["accuracy"])
        logger.info(f"  Best model: {best_key}")

        return outcome_results, (x_wo, x_w, y)

    def run_stacked_ensemble(self, outcome_name, outcome_series, feature_df, interaction_pairs=None, n_interactions=15):
        logger.info(f"\nStacked Ensemble: {outcome_name}")
        x_wo, x_w, y = _prepare_data(feature_df, outcome_series, interaction_pairs, n_interactions)
        if len(y.unique()) < 2:
            return {}, None

        x_tr, x_te, y_tr, y_te = train_test_split(x_w, y, test_size=0.25, random_state=self.random_state, stratify=y)
        multi = len(y.unique()) > 2

        estimators = [
            ("lr", LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight="balanced")),
            ("rf", RandomForestClassifier(n_estimators=150, random_state=self.random_state, n_jobs=-1, class_weight="balanced")),
        ]
        if len(y.unique()) == 2:
            estimators.append(("gb", GradientBoostingClassifier(n_estimators=100, random_state=self.random_state)))

        ensemble = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(max_iter=2000, random_state=self.random_state, class_weight="balanced"),
            cv=5,
        )
        ensemble.fit(x_tr, y_tr)
        y_pred = ensemble.predict(x_te)
        y_proba = ensemble.predict_proba(x_te) if hasattr(ensemble, "predict_proba") else None
        metrics = _eval_model(y_te, y_pred, y_proba)
        logger.info(f"  Ensemble: acc={metrics['accuracy']:.3f} f1={metrics.get('f1', 0):.3f}" +
                    (f" auc={metrics.get('roc_auc', 0):.3f}" if metrics.get("roc_auc") else ""))
        return {"model": ensemble, "metrics": metrics, "y_pred": y_pred, "y_test": y_te}, (x_tr.columns.tolist(), list(y.unique()))

    def run_shap_analysis(self, model, x_train):
        import shap
        logger.info("Running SHAP analysis")
        x_sample = x_train.sample(min(200, len(x_train)), random_state=self.random_state)
        try:
            explainer = shap.Explainer(model, x_sample)
            shap_values = explainer(x_sample)
            logger.info(f"  SHAP values shape: {shap_values.values.shape}")
            return shap_values
        except Exception as e:
            logger.warning(f"  SHAP failed: {e}")
            return None

    def run_survival_analysis(self, raw_df, feature_df=None):
        logger.info("\nSurvival Analysis (Kaplan-Meier + Cox PH)")
        raw = raw_df.copy()
        raw["exit_dt"] = pd.to_datetime(raw["ExitDate"], errors="coerce")
        raw["start_dt"] = pd.to_datetime(raw["StartDate"], errors="coerce")
        raw["is_terminated"] = raw["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
        raw["TenureDays"] = (raw["exit_dt"].fillna(pd.Timestamp.now()) - raw["start_dt"]).dt.days.clip(lower=0)

        terminated = raw[raw["is_terminated"] == 1].copy()
        logger.info(f"  Terminated: {len(terminated)} / Active: {len(raw) - len(terminated)}")

        kmf = KaplanMeierFitter()
        kmf.fit(raw["TenureDays"], event_observed=raw["is_terminated"])
        median_surv = kmf.median_survival_time_ if hasattr(kmf, 'median_survival_time_') else None
        logger.info(f"  KM median survival: {median_surv:.0f} days" if median_surv else "  KM median: N/A")

        cph = None
        cph_summary = None
        try:
            age_present = feature_df is not None and "Age" in feature_df
            if age_present:
                raw["Age"] = feature_df["Age"]
            cols = ["TenureDays", "is_terminated", "GenderCode"]
            if age_present:
                cols.insert(2, "Age")
            cph_df = raw[cols].copy().dropna()
            cph_df["GenderCode"] = (cph_df["GenderCode"] == "Male").astype(int)
            if feature_df is not None:
                extra_cols = ["SeniorityLevel", "IsManager", "IsExecutive"]
                for c in extra_cols:
                    if c in feature_df:
                        cph_df[c] = feature_df.loc[cph_df.index, c]
            if len(cph_df) > 100:
                cph = CoxPHFitter()
                cph.fit(cph_df, duration_col="TenureDays", event_col="is_terminated")
                cph_summary = cph.summary
                converged = getattr(cph, 'converged_', True)
                logger.info(f"  Cox PH converged: {converged} ({len(cph_df)} rows, {cph_df.columns[:-2].tolist()})")
        except Exception as e:
            logger.warning(f"  Cox PH failed: {e}")

        return kmf, cph, raw["TenureDays"]

    def run_all(self, raw_df, feature_df, interaction_results=None):
        logger.info("=" * 50)
        logger.info("Phase 4: Predictive Models")
        logger.info("=" * 50)

        if interaction_results is not None and len(interaction_results) > 0:
            top_pairs = [(r["feature_1"], r["feature_2"]) for _, r in interaction_results.head(20).iterrows()]
            logger.info(f"Top-20 interaction pairs: {top_pairs[:5]}...")
        else:
            top_pairs = []

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

        all_model_results = {}
        for name, outcome_series in outcome_defs.items():
            results, data = self.train_and_compare(name, outcome_series, feature_df, top_pairs)
            all_model_results[name] = {"results": results, "data": data}

        ensemble_results = {}
        for name in ["is_terminated", "PerfScore"]:
            if name in outcome_defs:
                ens, extra = self.run_stacked_ensemble(name, outcome_defs[name], feature_df, top_pairs)
                ensemble_results[name] = ens

        kmf, cph, survival_times = self.run_survival_analysis(raw_df, feature_df)

        rval = {
            "model_results": all_model_results,
            "ensemble_results": ensemble_results,
            "survival": {"kmf": kmf, "cph": cph, "survival_times": survival_times},
            "outcome_defs": outcome_defs,
            "feature_df": feature_df,
            "raw_df": raw_df,
        }

        logger.info(f"\n{'='*50}\nPhase 4 complete\n{'='*50}")
        return rval
