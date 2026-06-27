import streamlit as st
import pandas as pd
import plotly.express as px

from wf_analysis.analysis.diversity import DiversityAnalysis


def show(df=None):
    st.header("Diversity & Inclusion")

    if df is None:
        st.info("No data available.")
        return

    analysis = DiversityAnalysis()
    result = analysis.run(df)

    for fig in result.plots:
        st.plotly_chart(fig, use_container_width=True)

    chi = result.metrics.get("chi_square", {})
    if chi:
        st.subheader("Gender × JobFamily Independence Test")
        st.metric("Chi-Square p-value", f"{chi.get('p_value', 1):.4f}")
        st.caption("p > 0.05 suggests independence between gender and job family")

    st.subheader("Simpson Diversity Index")
    for key, val in result.metrics.items():
        if key.startswith("simpson_"):
            st.write(f"**{key}**")
            st.dataframe(pd.Series(val).to_frame("Index"), use_container_width=True)
