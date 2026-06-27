import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class AttritionAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Attrition Analysis")
        if "EmployeeStatus" not in df.columns:
            return result

        df = df.copy()
        df["is_attrited"] = df["EmployeeStatus"].str.lower() != "active"

        rate = df["is_attrited"].mean()
        result.metrics["attrition_rate"] = float(rate)
        result.metrics["attrition_count"] = int(df["is_attrited"].sum())
        result.metrics["total_count"] = len(df)

        if "DepartmentType" in df.columns:
            dept = (
                df.groupby("DepartmentType")["is_attrited"]
                .agg(["count", "sum"])
                .assign(rate=lambda x: x["sum"] / x["count"])
            )
            result.metrics["by_department"] = dept.to_dict("index")

        if "TerminationType" in df.columns:
            term = df[df["is_attrited"]]["TerminationType"].value_counts()
            result.metrics["termination_type"] = term.to_dict()

        if "GenderCode" in df.columns:
            gender = (
                df.groupby("GenderCode")["is_attrited"].mean()
            )
            result.metrics["attrition_by_gender"] = gender.to_dict()

        if "JobFamily" in df.columns:
            jf = (
                df.groupby("JobFamily")["is_attrited"]
                .agg(["count", "sum"])
                .assign(rate=lambda x: x["sum"] / x["count"])
                .sort_values("rate", ascending=False)
            )
            result.metrics["by_job_family"] = jf.to_dict("index")

        if "TenureYears" in df.columns:
            stayers = df[~df["is_attrited"]]["TenureYears"].mean()
            leavers = df[df["is_attrited"]]["TenureYears"].mean()
            result.metrics["avg_tenure_stayers"] = float(stayers) if pd.notna(stayers) else 0
            result.metrics["avg_tenure_leavers"] = float(leavers) if pd.notna(leavers) else 0

        fig = self.plot_attrition_rate(result)
        if fig:
            result.plots.append(fig)

        dept_fig = self.plot_dept_attrition(result)
        if dept_fig:
            result.plots.append(dept_fig)

        result.summary = (
            f"Overall attrition rate: {rate:.1%}. "
            f"{result.metrics['attrition_count']} employees attrited out of {result.metrics['total_count']}."
        )
        return result

    def plot_attrition_rate(self, result: AnalysisResult) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=round(result.metrics.get("attrition_rate", 0) * 100, 1),
            title={"text": "Attrition Rate (%)"},
            domain={"x": [0, 1], "y": [0, 1]},
            gauge={"axis": {"range": [0, 100]},
                   "bar": {"color": "#C73E1D" if result.metrics.get("attrition_rate", 0) > 0.3 else "#2E86AB"}},
        ))
        return fig

    def plot_dept_attrition(self, result: AnalysisResult) -> go.Figure | None:
        by_dept = result.metrics.get("by_department", {})
        if not by_dept:
            return None
        dept_df = pd.DataFrame(by_dept).T.reset_index()
        dept_df.columns = ["Department", "Total", "Attrited", "Rate"]
        fig = px.bar(
            dept_df, x="Department", y="Rate",
            title="Attrition Rate by Department",
            color="Rate", color_continuous_scale="Reds",
            text_auto=".0%",
        )
        return fig
