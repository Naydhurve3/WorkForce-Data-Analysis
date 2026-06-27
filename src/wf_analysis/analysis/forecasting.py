import pandas as pd
import plotly.express as px

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class ForecastingAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Workforce Planning & Forecasting")

        if "StartDate" in df.columns:
            df = df.copy()
            df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
            start_year = df["StartDate"].dt.year
            hires = start_year.value_counts().sort_index()
            result.metrics["hires_by_year"] = hires.to_dict()

        if "ExitDate" in df.columns:
            df["ExitDate"] = pd.to_datetime(df["ExitDate"], errors="coerce")
            exit_year = df["ExitDate"].dt.year
            exits = exit_year.value_counts().sort_index()
            result.metrics["exits_by_year"] = exits.to_dict()

        if "is_attrited" in df.columns and "StartDate" in df.columns:
            yearly = df.groupby(start_year)["is_attrited"].agg(["count", "sum"])
            yearly["rate"] = yearly["sum"] / yearly["count"]
            yearly["trend"] = yearly["rate"].rolling(3, min_periods=1).mean()
            result.metrics["attrition_trend"] = yearly.to_dict("index")

        trend = result.metrics.get("attrition_trend", {})
        if trend:
            trend_df = pd.DataFrame(trend).T.reset_index()
            trend_df.columns = ["Year", "Count", "Attrited", "Rate", "Trend"]
            fig = px.line(
                trend_df, x="Year", y="Rate",
                title="Attrition Rate Trend",
                markers=True,
            )
            fig.add_scatter(
                x=trend_df["Year"], y=trend_df["Trend"],
                mode="lines", name="3-Year MA",
                line=dict(dash="dash", color="#C73E1D"),
            )
            result.plots.append(fig)

        result.summary = (
            f"Hires tracked across {len(result.metrics.get('hires_by_year', {}))} years. "
            f"Attrition trend available: {len(trend)} periods."
        )
        return result
