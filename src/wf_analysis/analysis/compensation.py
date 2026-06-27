import pandas as pd
import plotly.express as px

from wf_analysis.analysis.base import BaseAnalysis, AnalysisResult


class CompensationAnalysis(BaseAnalysis):
    def run(self, df: pd.DataFrame) -> AnalysisResult:
        result = AnalysisResult(summary="Compensation Analysis")

        for col in ["PayZone", "EmployeeClassificationType", "BusinessUnit"]:
            if col in df.columns:
                result.metrics[f"{col}_count"] = df[col].value_counts().to_dict()

        if "PayZone" in df.columns and "GenderCode" in df.columns:
            cross = pd.crosstab(df["PayZone"], df["GenderCode"], normalize="index")
            result.metrics["payzone_by_gender"] = cross.to_dict("index")

        if "PayZone" in df.columns and "DepartmentType" in df.columns:
            cross_dept = pd.crosstab(
                df["DepartmentType"], df["PayZone"], normalize="index"
            )
            result.metrics["payzone_by_dept"] = cross_dept.to_dict("index")

        if "PayZone" in df.columns and "JobFamily" in df.columns:
            cross_jf = pd.crosstab(
                df["JobFamily"], df["PayZone"], normalize="index"
            )
            result.metrics["payzone_by_jobfamily"] = cross_jf.to_dict("index")

        dz = result.metrics.get("PayZone_count", {})
        if dz:
            fig = px.pie(
                names=list(dz.keys()),
                values=list(dz.values()),
                title="Pay Zone Distribution",
            )
            result.plots.append(fig)

        result.summary = f"Pay zones: {', '.join(dz.keys()) if dz else 'N/A'}."
        return result
