import pandas as pd
import numpy as np

from wf_analysis.features.base import BaseFeatureTransformer


class DemographicTransformer(BaseFeatureTransformer):
    def __init__(self, age_bins=None, age_labels=None):
        self.age_bins = age_bins or [0, 29, 39, 49, 59, 69, 120]
        self.age_labels = age_labels or ["<30", "30s", "40s", "50s", "60s", "70+"]
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "DemographicTransformer":
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        today = pd.Timestamp.today()

        if "DOB" in df.columns:
            df["DOB"] = pd.to_datetime(df["DOB"], errors="coerce")
            df["Age"] = ((today - df["DOB"]).dt.days / 365.25).round(1)
            df["BirthYear"] = df["DOB"].dt.year

        if "Age" not in df.columns and "DOB" in df.columns and "Age" not in df.columns:
            df["Age"] = ((today - df["DOB"]).dt.days / 365.25).round(1)

        if "Age" in df.columns:
            df["AgeGroup"] = pd.cut(
                df["Age"], bins=self.age_bins, labels=self.age_labels, right=False
            )
            age_nan = df["BirthYear"].isna() if "BirthYear" in df.columns else True
            if "BirthYear" in df.columns:
                df.loc[age_nan, "BirthYear"] = (today.year - df.loc[age_nan, "Age"]).round(0)
            else:
                df["BirthYear"] = today.year - df["Age"]
            df["BirthYear"] = df["BirthYear"].fillna(0).astype(int)
            by = df["BirthYear"]
            conditions = [
                by <= 1945,
                (by >= 1946) & (by <= 1964),
                (by >= 1965) & (by <= 1980),
                (by >= 1981) & (by <= 1996),
                by >= 1997,
            ]
            choices = ["Silent", "Boomer", "GenX", "Millennial", "GenZ"]
            df["Generation"] = np.select(conditions, choices, default="Unknown")

        if "ExitDate" in df.columns:
            df["IsActive"] = df["ExitDate"].isna()
        else:
            df["IsActive"] = True

        if "StartDate" in df.columns:
            df["StartDate"] = pd.to_datetime(df["StartDate"], errors="coerce")
            exit_date = pd.to_datetime(df.get("ExitDate", pd.NaT), errors="coerce")
            end = exit_date.fillna(today)
            df["TenureDays"] = (end - df["StartDate"]).dt.days
            df["TenureYears"] = (df["TenureDays"] / 365.25).round(1)
            bins = [-1, 365, 1095, 1825, 3650, 999999]
            labels = ["<1yr", "1-3yr", "3-5yr", "5-10yr", "10+yr"]
            df["TenureBucket"] = pd.cut(
                df["TenureDays"], bins=bins, labels=labels, right=True
            )

        return df
