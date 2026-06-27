import pandas as pd
import numpy as np
import plotly.express as px

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class PerformanceAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Performance Analytics")

        if "Performance Score" in df.columns:
            dist = df["Performance Score"].value_counts().to_dict()
            result.metrics["score_distribution"] = dist
            pct = df["Performance Score"].value_counts(normalize=True).to_dict()
            result.metrics["score_pct"] = pct

        if "Current Employee Rating" in df.columns:
            result.metrics["avg_rating"] = float(df["Current Employee Rating"].mean())
            result.metrics["median_rating"] = float(df["Current Employee Rating"].median())
            result.metrics["rating_std"] = float(df["Current Employee Rating"].std())

        if "Performance Score" in df.columns and "Current Employee Rating" in df.columns:
            rating_by_score = (
                df.groupby("Performance Score")["Current Employee Rating"]
                .agg(["mean", "count", "std"])
                .to_dict("index")
            )
            result.metrics["rating_by_score"] = rating_by_score

        pip_mask = df.get("Performance Score", "") == "PIP"
        result.metrics["pip_count"] = int(pip_mask.sum())
        result.metrics["pip_rate"] = float(pip_mask.mean())

        if "TenureYears" in df.columns and "Current Employee Rating" in df.columns:
            corr = df[["TenureYears", "Current Employee Rating"]].corr().iloc[0, 1]
            result.metrics["tenure_rating_corr"] = float(corr) if pd.notna(corr) else 0

        result.summary = (
            f"Average rating: {result.metrics.get('avg_rating', 0):.2f}. "
            f"PIP rate: {result.metrics.get('pip_rate', 0):.1%}. "
            f"Tenure-Rating correlation: {result.metrics.get('tenure_rating_corr', 0):.2f}."
        )

        score_dist = result.metrics.get("score_distribution", {})
        if score_dist:
            fig = px.bar(
                x=list(score_dist.keys()),
                y=list(score_dist.values()),
                title="Performance Score Distribution",
                labels={"x": "Score", "y": "Count"},
                color=list(score_dist.keys()),
                color_discrete_sequence=["#2E86AB", "#A23B72", "#F18F01", "#C73E1D"],
            )
            result.plots.append(fig)

        return result
