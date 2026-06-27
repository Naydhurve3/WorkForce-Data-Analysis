from wf_analysis.analysis.attrition import AttritionAnalysis
from wf_analysis.analysis.diversity import DiversityAnalysis
from wf_analysis.analysis.performance import PerformanceAnalysis
from wf_analysis.analysis.compensation import CompensationAnalysis
from wf_analysis.analysis.network import NetworkAnalysis
from wf_analysis.analysis.career_path import CareerPathAnalysis
from wf_analysis.analysis.forecasting import ForecastingAnalysis

import pandas as pd
import pytest


@pytest.fixture
def network_df():
    return pd.DataFrame({
        "EmpID": [101, 102, 103, 104, 105, 106, 107],
        "Title": ["CEO", "VP Eng", "VP Sales", "Engineer", "Engineer", "Sales Rep", "Sales Rep"],
        "Supervisor": ["", "CEO", "CEO", "VP Eng", "VP Eng", "VP Sales", "VP Sales"],
    })


@pytest.fixture
def career_df():
    return pd.DataFrame({
        "JobFunctionDescription": ["Engineering", "Engineering", "Sales", "Sales", "HR"],
        "Title": ["Junior Engineer", "Senior Engineer", "Sales Rep", "Sales Manager", "HR Coordinator"],
        "JobFamily": ["Engineering", "Engineering", "Sales", "Sales", "HR"],
        "TenureYears": [1.0, 5.0, 2.0, 8.0, 3.0],
    })


@pytest.fixture
def forecast_df():
    return pd.DataFrame({
        "StartDate": pd.to_datetime(["2020-01-01", "2020-06-15", "2021-03-01", "2022-07-01", "2023-01-15"]),
        "ExitDate": pd.to_datetime([None, "2021-06-15", None, "2022-12-01", None]),
        "is_attrited": [False, True, False, True, False],
    })


class TestAttritionAnalysis:
    def test_run_returns_rate(self, sample_df):
        a = AttritionAnalysis()
        result = a.run(sample_df)
        assert "attrition_rate" in result.metrics

    def test_plot_returns_figure(self, sample_df):
        a = AttritionAnalysis()
        result = a.run(sample_df)
        fig = a.plot(result)
        assert fig is not None or len(result.plots) > 0


class TestDiversityAnalysis:
    def test_run_returns_gender_dist(self, sample_df):
        d = DiversityAnalysis()
        result = d.run(sample_df)
        assert "GenderCode_distribution" in result.metrics


class TestPerformanceAnalysis:
    def test_run_returns_score_dist(self, sample_df):
        p = PerformanceAnalysis()
        result = p.run(sample_df)
        assert "score_distribution" in result.metrics


class TestCompensationAnalysis:
    def test_run_returns_payzone(self, sample_df):
        c = CompensationAnalysis()
        result = c.run(sample_df)
        assert "PayZone_count" in result.metrics


class TestNetworkAnalysis:
    def test_run_returns_total_nodes(self, network_df):
        n = NetworkAnalysis()
        result = n.run(network_df)
        assert result.metrics["total_nodes"] == 11

    def test_run_returns_total_edges(self, network_df):
        n = NetworkAnalysis()
        result = n.run(network_df)
        assert result.metrics["total_edges"] == 7

    def test_run_returns_span_of_control(self, network_df):
        n = NetworkAnalysis()
        result = n.run(network_df)
        soc = result.metrics["span_of_control"]
        for key in ["mean", "median", "std", "min", "max"]:
            assert key in soc

    def test_run_returns_top_influencers(self, network_df):
        n = NetworkAnalysis()
        result = n.run(network_df)
        assert isinstance(result.metrics["top_influencers"], list)
        assert len(result.metrics["top_influencers"]) > 0

    def test_run_handles_missing_columns(self, sample_df):
        n = NetworkAnalysis()
        result = n.run(sample_df.drop(columns=["Supervisor"]))
        assert result.metrics == {}

    def test_run_creates_directed_graph(self, network_df):
        n = NetworkAnalysis()
        result = n.run(network_df)
        assert result.metrics["total_edges"] == len(network_df)


class TestCareerPathAnalysis:
    def test_run_returns_job_function_dist(self, career_df):
        c = CareerPathAnalysis()
        result = c.run(career_df)
        assert "job_function_dist" in result.metrics

    def test_run_returns_top_titles(self, career_df):
        c = CareerPathAnalysis()
        result = c.run(career_df)
        assert "top_titles" in result.metrics
        assert len(result.metrics["top_titles"]) > 0

    def test_run_returns_job_family_dist(self, career_df):
        c = CareerPathAnalysis()
        result = c.run(career_df)
        assert "job_family_dist" in result.metrics

    def test_run_returns_title_similarity(self, career_df):
        c = CareerPathAnalysis()
        result = c.run(career_df)
        assert "title_similarity" in result.metrics

    def test_run_returns_avg_tenure_by_jobfamily(self, career_df):
        c = CareerPathAnalysis()
        result = c.run(career_df)
        assert "avg_tenure_by_jobfamily" in result.metrics

    def test_run_handles_missing_title(self, sample_df):
        c = CareerPathAnalysis()
        result = c.run(sample_df.drop(columns=["Title"]))
        assert "top_titles" not in result.metrics


class TestForecastingAnalysis:
    def test_run_returns_hires_by_year(self, forecast_df):
        f = ForecastingAnalysis()
        result = f.run(forecast_df)
        assert "hires_by_year" in result.metrics

    def test_run_returns_exits_by_year(self, forecast_df):
        f = ForecastingAnalysis()
        result = f.run(forecast_df)
        assert "exits_by_year" in result.metrics

    def test_run_returns_attrition_trend(self, forecast_df):
        f = ForecastingAnalysis()
        result = f.run(forecast_df)
        assert "attrition_trend" in result.metrics

    def test_run_handles_missing_startdate(self, sample_df):
        f = ForecastingAnalysis()
        result = f.run(sample_df.drop(columns=["StartDate"]))
        assert "hires_by_year" not in result.metrics
