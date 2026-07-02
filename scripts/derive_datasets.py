"""Derive all 10 analysis datasets from raw employee_data.csv."""

import os
import re
import json
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = "data/raw/employee_data.csv"
ANALYSIS_BASE = "data/analysis"


def parse_date(series, dayfirst=False):
    return pd.to_datetime(series, format="mixed", dayfirst=dayfirst, errors="coerce")


def tenure_days(start, end):
    """Calculate tenure in days from start to end (or today)."""
    today = pd.Timestamp.now()
    end = end.fillna(today)
    return (end - start).dt.days


def map_job_family(title_series):
    mapping = {
        "Director": "Leadership",
        "Manager": "Management",
        "Lead": "Management",
        "Chief": "Leadership",
        "VP": "Leadership",
        "Engineer": "Engineering",
        "Developer": "Engineering",
        "Architect": "Engineering",
        "Analyst": "Analytics",
        "Scientist": "Analytics",
        "Coordinator": "Operations",
        "Specialist": "Operations",
        "Associate": "Operations",
        "Representative": "Operations",
        "Agent": "Operations",
        "Administrator": "Administration",
        "Assistant": "Administration",
        "Clerk": "Administration",
        "Technician": "Technical",
        "Designer": "Creative",
    }

    def _map(title):
        if pd.isna(title):
            return "Unknown"
        for keyword, family in mapping.items():
            if keyword.lower() in title.lower():
                return family
        return "Other"

    return title_series.map(_map)


def map_seniority(title_series):
    levels = {
        "Director": 4,
        "VP": 5,
        "Chief": 5,
        "Manager": 3,
        "Lead": 3,
        "Senior": 3,
        "Junior": 1,
        "Associate": 1,
        "Intern": 0,
    }

    def _level(title):
        if pd.isna(title):
            return 1
        for keyword, lvl in levels.items():
            if keyword.lower() in title.lower():
                return lvl
        return 2

    return title_series.map(_level)


def map_region(state_series):
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


def map_division_group(division_series):
    group_map = {}
    for div in division_series.unique():
        if pd.isna(div):
            group_map[div] = "Unknown"
        elif any(k in str(div).lower() for k in ["corp", "exec", "admin", "hr"]):
            group_map[div] = "Corporate"
        elif any(k in str(div).lower() for k in ["sales", "market", "product", "customer"]):
            group_map[div] = "Revenue"
        elif any(k in str(div).lower() for k in ["eng", "tech", "data", "it", "r&d"]):
            group_map[div] = "Technology"
        elif any(k in str(div).lower() for k in ["ops", "oper", "logistic", "supply", "manufact"]):
            group_map[div] = "Operations"
        else:
            group_map[div] = "Other"
    return division_series.map(group_map)


def derive_01_profile(raw):
    """01 — Data Profile & Quality: metadata and quality metrics only."""
    df = raw.copy()
    profile = {
        "shape": list(df.shape),
        "columns": list(df.columns),
        "dtypes": {c: str(d) for c, d in df.dtypes.items()},
        "missing": {c: int(df[c].isnull().sum()) for c in df.columns},
        "missing_pct": {c: round(float(df[c].isnull().mean() * 100), 2) for c in df.columns},
        "unique_counts": {c: int(df[c].nunique()) for c in df.columns},
        "numeric_summary": {
            c: {
                "min": float(df[c].min()),
                "max": float(df[c].max()),
                "mean": round(float(df[c].mean()), 2),
                "std": round(float(df[c].std()), 2),
            }
            for c in df.select_dtypes(include=[np.number]).columns
        },
    }
    os.makedirs(f"{ANALYSIS_BASE}/01_data_profile", exist_ok=True)
    with open(f"{ANALYSIS_BASE}/01_data_profile/profile_report.json", "w") as f:
        json.dump(profile, f, indent=2, default=str)
    return profile


def derive_02_attrition(raw):
    """02 — Attrition Risk & Prediction dataset."""
    df = raw.copy()
    df["is_terminated"] = df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
    df["start_dt"] = parse_date(df["StartDate"])
    df["exit_dt"] = parse_date(df["ExitDate"])
    df["tenure_days"] = tenure_days(df["start_dt"], df["exit_dt"])
    df["tenure_years"] = (df["tenure_days"] / 365.25).round(1)
    df["age"] = (pd.Timestamp.now() - parse_date(df["DOB"])).dt.days / 365.25
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)
    df["job_family"] = map_job_family(df["Title"])
    df["seniority_level"] = map_seniority(df["Title"])
    df["region"] = map_region(df["State"])
    df["division_group"] = map_division_group(df["Division"])
    df["perf_encoded"] = df["Performance Score"].map({
        "PIP": 1, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4
    }).fillna(0).astype(int)

    keep = [
        "EmpID", "is_terminated", "tenure_days", "tenure_years", "age", "age_group",
        "job_family", "seniority_level", "region", "division_group",
        "GenderCode", "RaceDesc", "MaritalDesc", "DepartmentType",
        "BusinessUnit", "PayZone", "EmployeeType", "perf_encoded",
        "Current Employee Rating", "LocationCode", "StartDate", "ExitDate",
        "TerminationType", "TerminationDescription",
    ]
    out = df[keep].copy()
    out["GenderCode"] = out["GenderCode"].astype(str)
    out["RaceDesc"] = out["RaceDesc"].astype(str)
    out.to_parquet(f"{ANALYSIS_BASE}/02_attrition/dataset.parquet", index=False)
    return out


def derive_03_compensation(raw):
    """03 — Compensation Equity dataset (active employees only)."""
    df = raw.copy()
    df["is_active"] = df["EmployeeStatus"].str.lower() == "active"
    df["start_dt"] = parse_date(df["StartDate"])
    df["tenure_days"] = (pd.Timestamp.now() - df["start_dt"]).dt.days
    df["tenure_years"] = (df["tenure_days"] / 365.25).round(1)
    df["job_family"] = map_job_family(df["Title"])
    df["seniority_level"] = map_seniority(df["Title"])
    df["department_type"] = df["DepartmentType"]
    df["gender_code"] = df["GenderCode"].astype(str)
    df["race_desc"] = df["RaceDesc"].astype(str)
    df["pay_zone_encoded"] = df["PayZone"].map({"Zone A": 1, "Zone B": 2, "Zone C": 3}).fillna(0).astype(int)
    df["perf_encoded"] = df["Performance Score"].map({
        "PIP": 1, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4
    }).fillna(0).astype(int)

    active = df[df["is_active"]].copy()
    keep = [
        "EmpID", "gender_code", "race_desc", "PayZone", "pay_zone_encoded",
        "department_type", "BusinessUnit", "Division", "job_family",
        "seniority_level", "tenure_days", "tenure_years",
        "Current Employee Rating", "perf_encoded", "LocationCode",
        "MaritalDesc", "is_active",
    ]
    out = active[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/03_compensation/dataset.parquet", index=False)
    return out


def derive_04_performance(raw):
    """04 — Performance Drivers dataset (employees with ratings)."""
    df = raw.copy()
    df["has_rating"] = df["Current Employee Rating"].notna()
    df["is_active"] = df["EmployeeStatus"].str.lower() == "active"
    df["start_dt"] = parse_date(df["StartDate"])
    df["tenure_days"] = (pd.Timestamp.now() - df["start_dt"]).dt.days
    df["tenure_years"] = (df["tenure_days"] / 365.25).round(1)
    df["job_family"] = map_job_family(df["Title"])
    df["perf_encoded"] = df["Performance Score"].map({
        "PIP": 1, "Needs Improvement": 2, "Fully Meets": 3, "Exceeds": 4
    }).fillna(0).astype(int)
    df["high_performer"] = (df["perf_encoded"] >= 4).astype(int)
    df["seniority_level"] = map_seniority(df["Title"])
    df["age"] = (pd.Timestamp.now() - parse_date(df["DOB"])).dt.days / 365.25
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)

    rated = df[df["has_rating"]].copy()
    keep = [
        "EmpID", "Current Employee Rating", "perf_encoded", "high_performer",
        "tenure_days", "tenure_years", "age", "age_group",
        "job_family", "seniority_level", "DepartmentType", "GenderCode",
        "RaceDesc", "PayZone", "EmployeeType", "LocationCode",
        "MaritalDesc",
    ]
    out = rated[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/04_performance/dataset.parquet", index=False)
    return out


def derive_05_career(raw):
    """05 — Career Path & Mobility dataset."""
    df = raw.copy()
    df["start_dt"] = parse_date(df["StartDate"])
    df["exit_dt"] = parse_date(df["ExitDate"])
    df["tenure_days"] = tenure_days(df["start_dt"], df["exit_dt"])
    df["tenure_years"] = (df["tenure_days"] / 365.25).round(1)
    df["job_family"] = map_job_family(df["Title"])
    df["seniority_level"] = map_seniority(df["Title"])
    df["division_group"] = map_division_group(df["Division"])
    df["age"] = (pd.Timestamp.now() - parse_date(df["DOB"])).dt.days / 365.25
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)

    keep = [
        "EmpID", "Title", "job_family", "seniority_level", "DepartmentType",
        "Division", "division_group", "BusinessUnit", "tenure_days", "tenure_years",
        "age", "age_group", "GenderCode", "RaceDesc", "EmployeeStatus",
    ]
    out = df[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/05_career/dataset.parquet", index=False)
    return out


def derive_06_diversity(raw):
    """06 — Diversity & Inclusion dataset."""
    df = raw.copy()
    df["gender_code"] = df["GenderCode"].astype(str)
    df["race_desc"] = df["RaceDesc"].astype(str)
    df["marital_desc"] = df["MaritalDesc"].astype(str)
    df["department_type"] = df["DepartmentType"].astype(str)
    df["job_family"] = map_job_family(df["Title"])
    df["seniority_level"] = map_seniority(df["Title"])
    df["division_group"] = map_division_group(df["Division"])
    df["region"] = map_region(df["State"])

    keep = [
        "EmpID", "gender_code", "race_desc", "marital_desc", "department_type",
        "job_family", "seniority_level", "division_group", "region",
        "BusinessUnit", "LocationCode", "EmployeeType",
    ]
    out = df[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/06_diversity/dataset.parquet", index=False)
    return out


def derive_07_network(raw):
    """07 — Org Network & Span of Control dataset."""
    df = raw.copy()
    df["has_supervisor"] = df["Supervisor"].notna() & (df["Supervisor"] != "")
    df["job_family"] = map_job_family(df["Title"])
    df["seniority_level"] = map_seniority(df["Title"])

    keep = [
        "EmpID", "Supervisor", "Title", "job_family", "seniority_level",
        "DepartmentType", "Division", "BusinessUnit", "LocationCode",
        "has_supervisor",
    ]
    out = df[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/07_network/dataset.parquet", index=False)
    return out


def derive_08_forecast(raw):
    """08 — Workforce Forecasting dataset (active employees)."""
    df = raw.copy()
    df["is_active"] = df["EmployeeStatus"].str.lower() == "active"
    df["start_dt"] = parse_date(df["StartDate"])
    df["exit_dt"] = parse_date(df["ExitDate"])
    df["tenure_days"] = tenure_days(df["start_dt"], df["exit_dt"])
    df["tenure_years"] = (df["tenure_days"] / 365.25).round(1)
    df["age"] = (pd.Timestamp.now() - parse_date(df["DOB"])).dt.days / 365.25
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 25, 35, 45, 55, 120], labels=["<25", "25-34", "35-44", "45-54", "55+"]
    ).astype(str)
    df["retirement_risk"] = (df["age"].fillna(0) >= 55).astype(int)
    df["job_family"] = map_job_family(df["Title"])
    df["division_group"] = map_division_group(df["Division"])

    active = df[df["is_active"]].copy()
    keep = [
        "EmpID", "is_active", "tenure_days", "tenure_years", "age", "age_group",
        "retirement_risk", "job_family", "division_group", "DepartmentType",
        "BusinessUnit", "GenderCode", "RaceDesc", "Title", "LocationCode",
    ]
    out = active[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/08_forecast/dataset.parquet", index=False)
    return out


def derive_09_exit_nlp(raw):
    """09 — Exit Analysis & NLP dataset (terminated employees only)."""
    df = raw.copy()
    df["is_terminated"] = df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
    terminated = df[df["is_terminated"] == 1].copy()

    terminated["cleaned_desc"] = terminated["TerminationDescription"].fillna("").str.lower()
    terminated["cleaned_desc"] = terminated["cleaned_desc"].str.replace(r"[^a-z\s]", "", regex=True)
    terminated["cleaned_desc"] = terminated["cleaned_desc"].str.strip()
    terminated["desc_length"] = terminated["cleaned_desc"].str.len()
    terminated["word_count"] = terminated["cleaned_desc"].str.split().str.len()

    terminated["start_dt"] = parse_date(terminated["StartDate"])
    terminated["exit_dt"] = parse_date(terminated["ExitDate"])
    terminated["tenure_days"] = tenure_days(terminated["start_dt"], terminated["exit_dt"])
    terminated["tenure_years"] = (terminated["tenure_days"] / 365.25).round(1)
    terminated["job_family"] = map_job_family(terminated["Title"])
    terminated["division_group"] = map_division_group(terminated["Division"])

    keep = [
        "EmpID", "TerminationType", "TerminationDescription", "cleaned_desc",
        "desc_length", "word_count", "tenure_days", "tenure_years",
        "job_family", "division_group", "DepartmentType", "GenderCode",
        "RaceDesc", "MaritalDesc", "Title", "StartDate", "ExitDate",
    ]
    out = terminated[keep].copy()
    out.to_parquet(f"{ANALYSIS_BASE}/09_exit_nlp/dataset.parquet", index=False)
    return out


def derive_10_integrated(raw):
    """10 — Integrated Strategy dataset (merged from all 9)."""
    if not os.path.exists(f"{ANALYSIS_BASE}/02_attrition/dataset.parquet"):
        print("Warning: derived datasets not found. Run individual derivations first.")
        # Still create a basic version from raw
        df = raw.copy()
        df["is_terminated"] = df["EmployeeStatus"].str.lower().str.contains("terminat").astype(int)
        df["is_active"] = df["EmployeeStatus"].str.lower() == "active"
        df["job_family"] = map_job_family(df["Title"])
        df["division_group"] = map_division_group(df["Division"])
        df["risk_composite_score"] = 0

        out = df[["EmpID", "is_terminated", "is_active", "job_family",
                   "division_group", "DepartmentType", "GenderCode",
                   "RaceDesc", "risk_composite_score"]].copy()
        out.to_parquet(f"{ANALYSIS_BASE}/10_integrated/dataset.parquet", index=False)
        return out

    d02 = pd.read_parquet(f"{ANALYSIS_BASE}/02_attrition/dataset.parquet")
    d03 = pd.read_parquet(f"{ANALYSIS_BASE}/03_compensation/dataset.parquet")
    d04 = pd.read_parquet(f"{ANALYSIS_BASE}/04_performance/dataset.parquet")
    d05 = pd.read_parquet(f"{ANALYSIS_BASE}/05_career/dataset.parquet")
    d06 = pd.read_parquet(f"{ANALYSIS_BASE}/06_diversity/dataset.parquet")
    d07 = pd.read_parquet(f"{ANALYSIS_BASE}/07_network/dataset.parquet")
    d08 = pd.read_parquet(f"{ANALYSIS_BASE}/08_forecast/dataset.parquet")
    d09 = pd.read_parquet(f"{ANALYSIS_BASE}/09_exit_nlp/dataset.parquet")

    merged = d02[["EmpID", "is_terminated", "tenure_years", "age",
                   "age_group", "job_family", "region"]].copy()

    # Add flags from each analysis
    merged["has_comp_data"] = merged["EmpID"].isin(d03["EmpID"]).astype(int)
    merged["has_perf_data"] = merged["EmpID"].isin(d04["EmpID"]).astype(int)
    merged["has_network_data"] = merged["EmpID"].isin(d07["EmpID"]).astype(int)
    merged["is_active"] = (~merged["EmpID"].isin(d09["EmpID"])).astype(int)

    # Composite risk score (simple version)
    risk_cols = []
    if "is_terminated" in merged.columns:
        risk_cols.append(merged["is_terminated"])
    if len(risk_cols) > 0:
        merged["risk_composite_score"] = pd.concat(risk_cols, axis=1).mean(axis=1).round(2)
    else:
        merged["risk_composite_score"] = 0.0

    merged.to_parquet(f"{ANALYSIS_BASE}/10_integrated/dataset.parquet", index=False)
    return merged


DERIVE_FUNCTIONS = {
    1: ("01_data_profile", derive_01_profile),
    2: ("02_attrition", derive_02_attrition),
    3: ("03_compensation", derive_03_compensation),
    4: ("04_performance", derive_04_performance),
    5: ("05_career", derive_05_career),
    6: ("06_diversity", derive_06_diversity),
    7: ("07_network", derive_07_network),
    8: ("08_forecast", derive_08_forecast),
    9: ("09_exit_nlp", derive_09_exit_nlp),
    10: ("10_integrated", derive_10_integrated),
}


def derive_all():
    """Derive all 10 datasets."""
    print("=" * 60)
    print("  Workforce Data Analysis — Derived Dataset Generator")
    print("=" * 60)

    raw = pd.read_csv(RAW_PATH)
    print(f"\nLoaded raw data: {raw.shape[0]} rows, {raw.shape[1]} columns\n")

    results = {}
    for num in sorted(DERIVE_FUNCTIONS.keys()):
        name, func = DERIVE_FUNCTIONS[num]
        print(f"  [{num:02d}] Deriving {name}...", end=" ")
        result = func(raw)
        if isinstance(result, dict):
            print(f"done (profile saved)")
        elif isinstance(result, pd.DataFrame):
            print(f"done ({len(result):,} rows, {len(result.columns)} cols)")
        else:
            print(f"done")
        results[num] = result

    print("\n" + "=" * 60)
    print("  All derived datasets generated successfully.")
    print(f"  Location: {ANALYSIS_BASE}/{{nn}}_{{name}}/")
    print("=" * 60)
    return results


if __name__ == "__main__":
    derive_all()
