"""Phase 4: Predictive Models — LR→RF→XGB→Ensemble, WITH vs WITHOUT interactions + SHAP + Survival."""

import sys
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from loguru import logger

from wf_analysis.data.loader import DataLoader
from wf_analysis.interaction.config import InteractionConfig
from wf_analysis.interaction.features import FeatureEngineer
from wf_analysis.interaction.models import InteractionModeler
from wf_analysis.interaction.figures import EDAFigureFactory


def main():
    logger.info("=" * 60)
    logger.info("  Phase 4: Predictive Models — WITH vs WITHOUT Interactions")
    logger.info("=" * 60)

    cfg = InteractionConfig()

    raw_df = DataLoader.load(cfg.raw_path, validate=False)
    engineer = FeatureEngineer(cfg)
    feature_df = engineer.compute_all(raw_df)

    interaction_results = None
    ir_path = Path(cfg.output_dir) / "interaction_results.parquet"
    if ir_path.exists():
        interaction_results = pd.read_parquet(ir_path)
        logger.info(f"Loaded {len(interaction_results)} interaction results")
    else:
        logger.warning("No interaction results found — running without interaction features")

    modeler = InteractionModeler(cfg, cv=True)
    rval = modeler.run_all(raw_df, feature_df, interaction_results)

    figures = EDAFigureFactory(cfg)

    paths = []

    p1 = figures.figure_24_model_comparison(rval["model_results"])
    paths.append(p1); logger.info(f"Figure 24: Model comparison -> {p1}")

    attrition_key = "is_terminated"
    # Fix figure_28 to accept x_test by storing it during training
    for oname in rval["model_results"]:
        for key, res in rval["model_results"][oname]["results"].items():
            if "XGB" in key and hasattr(res["model"], "feature_importances_"):
                try:
                    _, x_w, _ = rval["model_results"][oname]["data"]
                    if x_w is not None:
                        res["_x_test"] = x_w.sample(min(500, len(x_w)), random_state=42)
                except Exception:
                    pass

    for oname in rval["model_results"]:
        or_ = rval["model_results"][oname]["results"]
        p2 = figures.figure_25_coefficient_shift(or_)
        paths.append(p2); logger.info(f"Figure 25: Coeff shift ({oname}) -> {p2}")
        break

    for oname in rval["model_results"]:
        or_ = rval["model_results"][oname]["results"]
        p3 = figures.figure_26_rf_importance(or_)
        paths.append(p3); logger.info(f"Figure 26: RF importance ({oname}) -> {p3}")
        p4 = figures.figure_27_importance_delta(or_)
        paths.append(p4); logger.info(f"Figure 27: Importance delta ({oname}) -> {p4}")
        p5 = figures.figure_28_xgb_roc(or_)
        paths.append(p5); logger.info(f"Figure 28: XGB performance ({oname}) -> {p5}")
        break

    attrition_results = rval["model_results"].get(attrition_key, {}).get("results", {})
    shap_values = None
    x_train_sample = None
    w_key = next((k for k in attrition_results if "RF" in k and "with" in k), None)
    if w_key and attrition_results[w_key].get("model") is not None:
        try:
            _, x_w, _ = rval["model_results"][attrition_key]["data"]
            if x_w is not None:
                x_train_sample = x_w.sample(min(200, len(x_w)), random_state=42)
                import shap
                explainer = shap.TreeExplainer(attrition_results[w_key]["model"], x_train_sample)
                shap_values = explainer(x_train_sample, check_additivity=False)
                logger.info(f"SHAP analysis complete: {shap_values.values.shape}")
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    feature_names = x_train_sample.columns.tolist() if x_train_sample is not None else []
    p6 = figures.figure_29_shap_summary(shap_values, feature_names)
    paths.append(p6); logger.info(f"Figure 29: SHAP summary -> {p6}")

    top_feat = feature_names[0] if feature_names else "Age"
    p7 = figures.figure_30_shap_dependence(shap_values, feature_df, feature_names, top_feature=top_feat)
    paths.append(p7); logger.info(f"Figure 30: SHAP dependence -> {p7}")

    p8 = figures.figure_31_shap_interaction(shap_values, feature_names)
    paths.append(p8); logger.info(f"Figure 31: SHAP interaction -> {p8}")

    p9 = figures.figure_32_ensemble_cm(rval["ensemble_results"].get(attrition_key, {}))
    paths.append(p9); logger.info(f"Figure 32: Ensemble CM -> {p9}")

    kmf = rval["survival"]["kmf"]
    survival_times = rval["survival"]["survival_times"]
    p10 = figures.figure_33_km_survival(kmf, survival_times)
    paths.append(p10); logger.info(f"Figure 33: KM survival -> {p10}")

    p11 = figures.figure_34_cox_hazard(rval["survival"]["cph"])
    paths.append(p11); logger.info(f"Figure 34: Cox PH -> {p11}")

    p12 = figures.figure_35_model_table(rval["model_results"])
    paths.append(p12); logger.info(f"Figure 35: Model table -> {p12}")

    p13 = figures.figure_36_modeling_dashboard(rval["model_results"], rval["ensemble_results"], shap_values is not None)
    paths.append(p13); logger.info(f"Figure 36: Modeling dashboard -> {p13}")

    summary = {"outcomes": list(rval["model_results"].keys()), "models_trained": len(paths)}
    with open(f"{cfg.output_dir}/modeling_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    serializable = {}
    for oname, odata in rval["model_results"].items():
        results_ser = {}
        for key, res in odata["results"].items():
            m = res.get("metrics", {})
            fi = res.get("feature_importance", {})
            if isinstance(fi, dict):
                fi = {k: float(v) for k, v in fi.items()}
            mc = res.get("metrics_cv", {})
            cm = m.get("cm", None)
            results_ser[key] = {
                "metrics": {k: float(v) if isinstance(v, (int, float)) else v for k, v in m.items() if k != "cm"},
                "metrics_cv": {k: float(v) if isinstance(v, (int, float)) else v for k, v in mc.items()} if mc else {},
                "cm": cm,
                "feature_importance": fi,
                "coefficients": res.get("coefficients"),
                "feature_names": res.get("feature_names", []),
            }
            if "y_test" in res and "y_pred" in res:
                y_true = np.array(res["y_test"])
                y_pred = np.array(res["y_pred"])
                y_proba_val = None
                if "y_proba" in res and res["y_proba"] is not None:
                    y_proba_val = np.array(res["y_proba"])
                elif hasattr(res.get("model"), "predict_proba") and "x_test" in res:
                    try:
                        y_proba_val = res["model"].predict_proba(res["x_test"])
                    except Exception:
                        pass
                if y_proba_val is not None and y_proba_val.ndim == 2 and y_proba_val.shape[1] == 2:
                    fpr, tpr, _ = roc_curve(y_true, y_proba_val[:, 1])
                    results_ser[key]["roc_fpr"] = fpr.tolist()
                    results_ser[key]["roc_tpr"] = tpr.tolist()
        serializable[oname] = results_ser

    with open(f"{cfg.output_dir}/model_results.json", "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    logger.info(f"  Model results saved to {cfg.output_dir}/model_results.json")

    logger.info("=" * 60)
    logger.info(f"  Phase 4 Complete: {len(paths)} figures generated")
    logger.info(f"  SHAP: {'Yes' if shap_values is not None else 'No'}")
    logger.info(f"  Survival: KM + Cox PH")
    logger.info("=" * 60)
    return paths


if __name__ == "__main__":
    main()
