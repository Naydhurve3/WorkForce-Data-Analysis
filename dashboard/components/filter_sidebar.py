import streamlit as st
import pandas as pd


def render_filters(df: pd.DataFrame) -> dict:
    st.sidebar.markdown("### Global Filters")

    filters = {}

    if "JobFamily" in df.columns:
        options = sorted(df["JobFamily"].dropna().unique())
        filters["job_families"] = st.sidebar.multiselect(
            "Job Family", options, default=options
        )

    if "DepartmentType" in df.columns:
        options = sorted(df["DepartmentType"].dropna().unique())
        filters["departments"] = st.sidebar.multiselect(
            "Department", options, default=options
        )

    if "Region" in df.columns:
        options = sorted(df["Region"].dropna().unique())
        filters["regions"] = st.sidebar.multiselect(
            "Region", options, default=options
        )

    if "GenderCode" in df.columns:
        options = sorted(df["GenderCode"].dropna().unique())
        filters["genders"] = st.sidebar.multiselect(
            "Gender", options, default=options
        )

    if "Age" in df.columns:
        min_age, max_age = int(df["Age"].min()), int(df["Age"].max())
        filters["age_range"] = st.sidebar.slider(
            "Age Range", min_age, max_age, (min_age, max_age)
        )

    st.sidebar.markdown("---")

    if st.sidebar.button("Reset Filters"):
        st.rerun()

    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    filtered = df.copy()
    if "job_families" in filters and filters["job_families"]:
        if "JobFamily" in filtered.columns:
            filtered = filtered[filtered["JobFamily"].isin(filters["job_families"])]
    if "departments" in filters and filters["departments"]:
        if "DepartmentType" in filtered.columns:
            filtered = filtered[filtered["DepartmentType"].isin(filters["departments"])]
    if "regions" in filters and filters["regions"]:
        if "Region" in filtered.columns:
            filtered = filtered[filtered["Region"].isin(filters["regions"])]
    if "genders" in filters and filters["genders"]:
        if "GenderCode" in filtered.columns:
            filtered = filtered[filtered["GenderCode"].isin(filters["genders"])]
    if "age_range" in filters:
        if "Age" in filtered.columns:
            lo, hi = filters["age_range"]
            filtered = filtered[filtered["Age"].between(lo, hi)]
    return filtered
