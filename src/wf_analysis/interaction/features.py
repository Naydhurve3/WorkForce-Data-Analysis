import pandas as pd
import numpy as np
from loguru import logger



def _map_job_family(title_series):
    mapping = {
        "Director": "Leadership", "Manager": "Management", "Lead": "Management",
        "Chief": "Leadership", "VP": "Leadership", "Engineer": "Engineering",
        "Developer": "Engineering", "Architect": "Engineering", "Analyst": "Analytics",
        "Scientist": "Analytics", "Coordinator": "Operations", "Specialist": "Operations",
        "Associate": "Operations", "Representative": "Operations", "Agent": "Operations",
        "Administrator": "Administration", "Assistant": "Administration", "Clerk": "Administration",
        "Technician": "Technical", "Designer": "Creative",
    }
    def _map(v):
        if pd.isna(v):
            return "Unknown"
        for kw, fam in mapping.items():
            if kw.lower() in str(v).lower():
                return fam
        return "Other"
    return title_series.map(_map)


def _map_seniority(title_series):
    def _level(v):
        if pd.isna(v):
            return 1
        v = str(v)
        if any(k in v for k in ["Director", "VP", "Chief"]):
            return 4
        if any(k in v for k in ["Manager", "Lead", "Senior", "Sr"]):
            return 3
        if any(k in v for k in ["Junior", "Jr", "Associate", "Intern"]):
            return 1
        return 2
    return title_series.map(_level)


def _map_division_group(division_series):
    def _map(d):
        if pd.isna(d):
            return "Unknown"
        d = str(d).lower()
        if any(k in d for k in ["corp", "exec", "admin", "hr"]):
            return "Corporate"
        if any(k in d for k in ["sales", "market", "product", "customer"]):
            return "Revenue"
        if any(k in d for k in ["eng", "tech", "data", "it", "r&d"]):
            return "Technology"
        if any(k in d for k in ["ops", "oper", "logistic", "supply", "manufact"]):
            return "Operations"
        return "Other"
    return division_series.map(_map)


def _map_region(state_series):
    region_map = {
        "CA": "West", "OR": "West", "WA": "West", "NV": "West", "AZ": "West",
        "CO": "West", "UT": "West", "AK": "West", "HI": "West",
        "NY": "Northeast", "MA": "Northeast", "NJ": "Northeast", "PA": "Northeast",
        "CT": "Northeast", "RI": "Northeast", "NH": "Northeast", "VT": "Northeast",
        "ME": "Northeast", "TX": "South", "FL": "South", "GA": "South",
        "NC": "South", "VA": "South", "TN": "South", "SC": "South",
        "AL": "South", "MS": "South", "LA": "South", "AR": "South", "OK": "South",
        "KY": "South", "WV": "South", "MD": "South", "DE": "South", "DC": "South",
        "IL": "Midwest", "OH": "Midwest", "MI": "Midwest", "IN": "Midwest",
        "WI": "Midwest", "MN": "Midwest", "IA": "Midwest", "MO": "Midwest",
        "KS": "Midwest", "NE": "Midwest", "SD": "Midwest", "ND": "Midwest",
    }
    return state_series.map(region_map).fillna("Other")


class FeatureEngineer:
    def __init__(self, config, random_state=42):
        self.cfg = config
        self.random_state = random_state

    def _compute_age_features(self, df):
        today = pd.Timestamp.now()
        dob = pd.to_datetime(df["DOB"], errors="coerce") if "DOB" in df.columns else pd.NaT
        age_years = (today - dob).dt.days / 365.25 if "DOB" in df.columns else np.nan
        df["Age"] = age_years.round(1).fillna(
            df["Current Employee Rating"] * 8 + 22
        )
        df["Generation"] = df["Age"].apply(
            lambda a: "GenZ" if a < 27 else "Millennial" if a < 42 else "GenX" if a < 58 else "Boomer" if a < 77 else "Silent"
        )
        return df

    def _compute_tenure_features(self, df):
        today = pd.Timestamp.now()
        end = df["_exit_dt"].fillna(today)
        df["TenureDays"] = (end - df["_start_dt"]).dt.days.clip(lower=0)
        df["TenureYears"] = (df["TenureDays"] / 365.25).round(1)
        df["IsLongTenure"] = (df["TenureYears"] >= 10).astype(int)
        avg_tenure = df["TenureYears"].mean()
        df["TenureVsAvg"] = (df["TenureYears"] - avg_tenure).round(1)
        return df

    def _compute_career_stage(self, df):
        bins = [0, 25, 35, 45, 55, 120]
        labels = ["Early", "Developing", "Mid", "Senior", "Late"]
        df["CareerStage"] = pd.cut(df["Age"], bins=bins, labels=labels, right=False).astype(str)
        return df

    def _compute_role_features(self, df):
        df["JobFamily"] = _map_job_family(df["Title"])
        df["SeniorityLevel"] = _map_seniority(df["Title"])
        df["IsExecutive"] = (df["SeniorityLevel"] >= 4).astype(int)
        df["IsManager"] = df["Title"].str.contains("Manager|Lead|Supervisor", na=False, regex=True).astype(int)
        df["IsIC"] = ((~df["IsManager"].astype(bool)) & (df["SeniorityLevel"] < 3)).astype(int)
        return df

    def _compute_org_features(self, df):
        sup = df["Supervisor"].fillna("").astype(str)
        sup_counts = sup[sup != ""].value_counts()
        df["SpanOfControl"] = df["FirstName"].str.cat(df["LastName"], sep=" ").map(sup_counts).fillna(0).astype(int)

        sup_map = dict(zip(
            df["FirstName"].str.cat(df["LastName"], sep=" "),
            sup
        ))
        def _org_depth(name, visited=None):
            if visited is None:
                visited = set()
            if name in visited or name not in sup_map or pd.isna(sup_map.get(name)):
                return 0
            visited.add(name)
            return 1 + _org_depth(sup_map[name], visited)

        employee_names = df["FirstName"].str.cat(df["LastName"], sep=" ")
        df["OrgLevel"] = employee_names.apply(lambda n: min(_org_depth(n), 20)).astype(int)
        df["OrgLevel"] = df["OrgLevel"].clip(upper=10)
        return df

    def _compute_geo_features(self, df):
        df["DivisionGroup"] = _map_division_group(df["Division"])
        df["Region"] = _map_region(df["State"])
        return df

    def _compute_diversity_features(self, df):
        df["GenderCode"] = df["GenderCode"].astype(str)
        df["RaceDesc"] = df["RaceDesc"].astype(str)

        dept_gender = df.groupby("DepartmentType")["GenderCode"].apply(
            lambda g: (g == "Male").mean()
        ).to_dict()
        df["DeptGenderRatio"] = df["DepartmentType"].map(dept_gender).fillna(0.5).round(3)

        dept_div = df.groupby("DepartmentType")["RaceDesc"].apply(
            lambda g: 1 - sum((g.value_counts(normalize=True) ** 2))
        ).to_dict()
        df["DeptDiversityScore"] = df["DepartmentType"].map(dept_div).fillna(0).round(3)

        df["IntersectionalID"] = (
            df["GenderCode"].str[0] + "_" +
            df["RaceDesc"].str[:3] + "_" +
            df["DepartmentType"].str[:4]
        )
        return df

    def _compute_date_features(self, df):
        df["StartYear"] = df["_start_dt"].dt.year.fillna(0).astype(int)
        df["StartQuarter"] = df["_start_dt"].dt.quarter.fillna(0).astype(int)
        df["ExitYear"] = df["_exit_dt"].dt.year.fillna(0).astype(int)
        df["ExitQuarter"] = df["_exit_dt"].dt.quarter.fillna(0).astype(int)
        return df

    def _compute_performance_features(self, df):
        perf_map = {"PIP": 1, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4}
        df["PerfScore"] = df["Performance Score"].map(perf_map).fillna(0).astype(int) if "Performance Score" in df.columns else 0
        return df

    def compute_all(self, df):
        logger.info("Engineering all 27 features")
        df = df.copy()
        df["_start_dt"] = pd.to_datetime(df["StartDate"], errors="coerce")
        df["_exit_dt"] = pd.to_datetime(df["ExitDate"], errors="coerce")
        df = self._compute_age_features(df)
        df = self._compute_tenure_features(df)
        df = self._compute_career_stage(df)
        df = self._compute_role_features(df)
        df = self._compute_org_features(df)
        df = self._compute_geo_features(df)
        df = self._compute_diversity_features(df)
        df = self._compute_date_features(df)
        df = self._compute_performance_features(df)
        df = df.drop(columns=["_start_dt", "_exit_dt"], errors="ignore")

        feature_cols = [
            "Age", "TenureDays", "TenureYears", "CareerStage", "IsLongTenure", "TenureVsAvg",
            "JobFamily", "SeniorityLevel", "DivisionGroup", "Region", "Generation",
            "IsExecutive", "IsManager", "IsIC",
            "SpanOfControl", "OrgLevel",
            "DeptGenderRatio", "DeptDiversityScore", "IntersectionalID",
            "StartYear", "StartQuarter", "ExitYear", "ExitQuarter",
            "PerfScore",
            "GenderCode", "RaceDesc", "DepartmentType",
        ]

        existing = [c for c in feature_cols if c in df.columns]
        logger.info(f"Engineered {len(existing)} features")
        return df[existing].copy()
