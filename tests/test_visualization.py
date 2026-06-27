import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from wf_analysis.analysis.base import AnalysisResult
from wf_analysis.visualization.theme import Theme
from wf_analysis.visualization.plots import PlotFactory
from wf_analysis.visualization.dashboards import DashboardBuilder
from wf_analysis.visualization.reports import ReportGenerator


class TestTheme:
    def test_set_style_runs_without_error(self):
        Theme.set_style()

    def test_get_palette_returns_primary(self):
        palette = Theme.get_palette("primary")
        assert palette == Theme.PRIMARY

    def test_get_palette_returns_diverging(self):
        palette = Theme.get_palette("diverging")
        assert palette == Theme.DIVERGING

    def test_get_palette_returns_categorical(self):
        palette = Theme.get_palette("categorical")
        assert palette == Theme.CATEGORICAL


class TestPlotFactory:
    def test_bar_chart_returns_figure(self):
        df = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
        fig = PlotFactory.bar_chart(df, x="x", y="y")
        assert isinstance(fig, plt.Figure)

    def test_box_plot_returns_figure(self):
        df = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
        fig = PlotFactory.box_plot(df, x="x", y="y")
        assert isinstance(fig, plt.Figure)

    def test_kde_plot_returns_figure(self):
        df = pd.DataFrame({"x": [1, 2, 3, 4]})
        fig = PlotFactory.kde_plot(df, x="x")
        assert isinstance(fig, plt.Figure)

    def test_heatmap_returns_figure(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]}, index=["x", "y"])
        fig = PlotFactory.heatmap(df)
        assert isinstance(fig, plt.Figure)

    def test_pie_chart_returns_figure(self):
        s = pd.Series({"A": 10, "B": 20, "C": 30})
        fig = PlotFactory.pie_chart(s)
        assert isinstance(fig, plt.Figure)


class TestDashboardBuilder:
    def test_build_grid_empty_list(self):
        fig = DashboardBuilder.build_grid([])
        assert isinstance(fig, plt.Figure)

    def test_build_grid_with_single_plot(self):
        df = pd.DataFrame({"x": ["A", "B"], "y": [1, 2]})
        single = PlotFactory.bar_chart(df, x="x", y="y")
        fig = DashboardBuilder.build_grid([single])
        assert isinstance(fig, plt.Figure)


class TestReportGenerator:
    def test_generate_html_returns_string_with_correct_structure(self, tmp_path):
        mock = AnalysisResult(summary="Test summary", metrics={"key": "value"})
        analyses = {"Test Analysis": mock}
        output = tmp_path / "report.html"
        html = ReportGenerator.generate_html(analyses, output_path=str(output))
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html
        assert "<h1>WorkForce Data Analysis Report</h1>" in html
        assert "Test Analysis" in html
        assert "Test summary" in html
