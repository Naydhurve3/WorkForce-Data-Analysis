import pandas as pd

from wf_analysis.features.base import BaseFeatureTransformer


class TemporalTransformer(BaseFeatureTransformer):
    def __init__(self, reference_date: str | None = None):
        self.reference_date = (
            pd.Timestamp(reference_date) if reference_date else pd.Timestamp.today()
        )
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "TemporalTransformer":
        self._fitted = True
        return self

    def _season(self, month: int) -> str:
        if month in (12, 1, 2):
            return "Winter"
        elif month in (3, 4, 5):
            return "Spring"
        elif month in (6, 7, 8):
            return "Summer"
        else:
            return "Fall"

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in ["StartDate", "ExitDate", "DOB"]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
                suffix = col.replace("Date", "")
                df[f"{suffix}Year"] = df[col].dt.year
                if suffix:
                    df[f"{suffix}Month"] = df[col].dt.month
                    df[f"{suffix}Quarter"] = df[col].dt.quarter
                    if suffix == "Start":
                        df["JoinSeason"] = df[col].dt.month.apply(self._season)
                    elif suffix == "Exit":
                        df["ExitSeason"] = df[col].dt.month.apply(self._season)

        if "DOBYear" in df.columns and "BirthYear" in df.columns:
            df["DOBYear"] = df["DOBYear"].fillna(df["BirthYear"])
        if "DOBMonth" in df.columns:
            df["DOBMonth"] = df["DOBMonth"].fillna(6)
        if "DOBQuarter" in df.columns:
            df["DOBQuarter"] = df["DOBQuarter"].fillna(2)

        if "StartDate" in df.columns and "ExitDate" in df.columns:
            exit_date = df["ExitDate"].fillna(self.reference_date)
            df["TenureDays"] = (exit_date - df["StartDate"]).dt.days
            df["TenureYears"] = (df["TenureDays"] / 365.25).round(1)
        elif "StartDate" in df.columns:
            df["TenureDays"] = (self.reference_date - df["StartDate"]).dt.days
            df["TenureYears"] = (df["TenureDays"] / 365.25).round(1)

        return df
