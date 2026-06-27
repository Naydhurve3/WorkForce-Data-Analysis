import numpy as np
import pandas as pd
import pytest

from wf_analysis.imputation.statistical import StatisticalImputer
from wf_analysis.imputation.predictive import PredictiveImputer
from wf_analysis.imputation.ensemble import EnsembleImputer
from wf_analysis.imputation.validator import ImputationValidator


class TestStatisticalImputer:
    def test_fit_and_impute(self):
        imp = StatisticalImputer(method="mean")
        df = pd.DataFrame({"Age": [1.0, 2.0, None, 4.0], "Group": ["A", "A", "A", "A"]})
        imp.fit(df, target_column="Age", feature_columns=[])
        result = imp.impute(df)
        assert result["Age"].iloc[2] == pytest.approx(7.0 / 3, rel=0.01)

    def test_median_imputation(self):
        imp = StatisticalImputer(method="median")
        df = pd.DataFrame({"Age": [1.0, 2.0, None, 10.0], "Group": ["A", "A", "A", "A"]})
        imp.fit(df, target_column="Age", feature_columns=[])
        result = imp.impute(df)
        assert result["Age"].iloc[2] == 2.0

    def test_mean_with_feature_columns(self):
        imp = StatisticalImputer(method="mean")
        df = pd.DataFrame({"Age": [1.0, 3.0, None, None], "Group": ["A", "B", "A", "B"]})
        imp.fit(df, target_column="Age", feature_columns=["Group"])
        result = imp.impute(df)
        assert result["Age"].iloc[2] == pytest.approx(1.0, rel=0.01)
        assert result["Age"].iloc[3] == pytest.approx(3.0, rel=0.01)

    def test_no_missing_data(self):
        imp = StatisticalImputer(method="mean")
        df = pd.DataFrame({"Age": [1.0, 2.0, 3.0], "Group": ["A", "A", "A"]})
        imp.fit(df, target_column="Age", feature_columns=[])
        result = imp.impute(df)
        assert result["Age"].tolist() == [1.0, 2.0, 3.0]

    def test_empty_feature_columns(self):
        imp = StatisticalImputer(method="median")
        df = pd.DataFrame({"Age": [1.0, None, 2.0]})
        imp.fit(df, target_column="Age", feature_columns=[])
        result = imp.impute(df)
        assert result["Age"].iloc[1] == 1.5


class TestPredictiveImputer:
    def test_rf_fit_and_impute(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = PredictiveImputer(model_type="rf", random_state=42)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        assert "r2" in imp.metrics
        assert "mae" in imp.metrics
        assert "rmse" in imp.metrics
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_gbm_fit_and_impute(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = PredictiveImputer(model_type="gbm", random_state=42)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        assert "r2" in imp.metrics
        assert "mae" in imp.metrics
        assert "rmse" in imp.metrics
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_all_numeric_features(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = PredictiveImputer(model_type="rf", random_state=42)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["EmpID", "LocationCode"])
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_all_categorical_features(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = PredictiveImputer(model_type="gbm", random_state=42)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["GenderCode", "State"])
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_no_missing_data(self, sample_df):
        df = sample_df.copy()
        imp = PredictiveImputer(model_type="rf", random_state=42)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        result = imp.impute(df)
        pd.testing.assert_series_equal(
            result["Current Employee Rating"], df["Current Employee Rating"]
        )


class TestEnsembleImputer:
    def test_default_params(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = EnsembleImputer()
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_custom_weights(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = EnsembleImputer(weights=[0.5, 0.3, 0.2])
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_custom_models_list(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        models = [("statistical", {"method": "mean"}), ("predictive", {"model_type": "gbm"})]
        imp = EnsembleImputer(models=models)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        assert len(imp._imputers) == 2
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_distribution_match_disabled(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = EnsembleImputer(use_distribution_match=False)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_distribution_match_enabled(self, sample_df):
        df = sample_df.copy()
        df.loc[0, "Current Employee Rating"] = None
        imp = EnsembleImputer(use_distribution_match=True)
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        result = imp.impute(df)
        assert not result["Current Employee Rating"].isna().any()

    def test_no_missing_data(self, sample_df):
        df = sample_df.copy()
        imp = EnsembleImputer()
        imp.fit(df, target_column="Current Employee Rating", feature_columns=["LocationCode"])
        result = imp.impute(df)
        pd.testing.assert_frame_equal(result, df)

    def test_unknown_model_type(self):
        imp = EnsembleImputer(models=[("knn", {})])
        df = pd.DataFrame({"Value": [1.0, 2.0, None], "Feature": ["A", "A", "A"]})
        with pytest.raises(ValueError, match="Unknown model type: knn"):
            imp.fit(df, target_column="Value", feature_columns=["Feature"])


class TestImputationValidator:
    def test_compare_distributions_returns_stats(self):
        orig = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        imp = pd.Series([1.1, 2.1, 2.9, 4.2, 5.1])
        result = ImputationValidator.compare_distributions(orig, imp, plot=False)
        assert "ks_statistic" in result
        assert "wasserstein_distance" in result

    def test_compare_distributions_with_plot(self):
        orig = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        imp = pd.Series([1.1, 2.1, 2.9, 4.2, 5.1])
        result = ImputationValidator.compare_distributions(orig, imp, plot=True)
        assert "plot" in result
        assert result["plot"] is not None

    def test_compare_distributions_empty_series(self):
        orig = pd.Series([], dtype=float)
        imp = pd.Series([1.0, 2.0, 3.0])
        result = ImputationValidator.compare_distributions(orig, imp, plot=False)
        assert "error" in result

    def test_full_report(self, sample_df):
        df_orig = sample_df.copy()
        df_imp = sample_df.copy()
        df_imp["Current Employee Rating"] = df_imp["Current Employee Rating"] + np.random.default_rng(42).normal(0, 0.1, size=len(df_imp))
        report = ImputationValidator.full_report(df_orig, df_imp, target_columns=["Current Employee Rating"])
        assert "Current Employee Rating" in report
        assert "ks_statistic" in report["Current Employee Rating"]
