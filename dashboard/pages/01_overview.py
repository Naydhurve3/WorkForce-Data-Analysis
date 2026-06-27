import streamlit as st
import pandas as pd


def show(df=None):
    st.header("Executive Overview")

    if df is not None:
        total = len(df)
        active = len(df[df.get("EmployeeStatus", "") == "Active"]) if "EmployeeStatus" in df.columns else "N/A"
        attrition_rate = (1 - active / total) if isinstance(active, int) and total > 0 else "N/A"
        avg_tenure = df["TenureYears"].mean() if "TenureYears" in df.columns else "N/A"
        total_str = f"{total:,}"
        active_str = f"{active:,}" if isinstance(active, int) else str(active)
        rate_str = f"{attrition_rate:.1%}" if isinstance(attrition_rate, float) else str(attrition_rate)
        tenure_str = f"{avg_tenure:.1f} yrs" if isinstance(avg_tenure, float) else str(avg_tenure)
    else:
        total_str = "3,000"
        active_str = "1,467"
        rate_str = "51.1%"
        tenure_str = "5.2 yrs"

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Headcount", total_str)
    with col2:
        st.metric("Attrition Rate", rate_str)
    with col3:
        st.metric("Avg Tenure", tenure_str)
    with col4:
        st.metric("Active Employees", active_str)

    st.markdown("---")
    if df is not None:
        st.subheader("Dataset Summary")
        st.dataframe(df.describe(include="all").T, use_container_width=True)
    else:
        st.subheader("Dataset Summary")
        st.info("Run `python scripts/run_pipeline.py` to process the raw data and populate this dashboard.")
