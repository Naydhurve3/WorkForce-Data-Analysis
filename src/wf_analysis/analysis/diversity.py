import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import plotly.express as px

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class DiversityAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Diversity & Inclusion Analysis")

        for col in ["GenderCode", "RaceDesc", "MaritalDesc"]:
            if col in df.columns:
                dist = df[col].value_counts(normalize=True).to_dict()
                result.metrics[f"{col}_distribution"] = dist

        if "Division" in df.columns and "GenderCode" in df.columns:
            cross = pd.crosstab(df["Division"], df["GenderCode"], normalize="index")
            result.metrics["gender_by_division"] = cross.to_dict("index")

        if "JobFamily" in df.columns and "GenderCode" in df.columns:
            cross_jf = pd.crosstab(df["JobFamily"], df["GenderCode"], normalize="index")
            result.metrics["gender_by_jobfamily"] = cross_jf.to_dict("index")

        if "GenderCode" in df.columns and "JobFamily" in df.columns:
            ct = pd.crosstab(df["GenderCode"], df["JobFamily"])
            chi2, p, dof, expected = chi2_contingency(ct)
            result.metrics["chi_square"] = {
                "statistic": float(chi2),
                "p_value": float(p),
                "degrees_of_freedom": int(dof),
                "independent": bool(p > 0.05),
            }

        for group_col in ["Division", "JobFamily", "DepartmentType"]:
            if group_col in df.columns and "GenderCode" in df.columns:
                simpson = {}
                for group, sub in df.groupby(group_col):
                    props = sub["GenderCode"].value_counts(normalize=True).values
                    simpson[group] = float(1 - np.sum(props**2))
                result.metrics[f"simpson_gender_by_{group_col}"] = simpson

        figs = self._make_plots(result, df)
        result.plots.extend(figs)

        result.summary = (
            f"Gender distribution analyzed across {len(result.metrics.get('gender_by_division', {}))} divisions. "
            f"Chi-square test p={result.metrics.get('chi_square', {}).get('p_value', 0):.4f}."
        )
        return result

    def _make_plots(self, result: AnalysisResult, df: pd.DataFrame) -> list:
        figs = []
        for col in ["GenderCode", "RaceDesc"]:
            dist = result.metrics.get(f"{col}_distribution", {})
            if dist:
                fig = px.pie(
                    names=list(dist.keys()),
                    values=list(dist.values()),
                    title=f"Distribution by {col}",
                )
                figs.append(fig)
        return figs
