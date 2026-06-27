import streamlit as st
import pandas as pd

from wf_analysis.analysis.network import NetworkAnalysis


def show(df=None):
    st.header("Organizational Network")

    if df is None:
        st.info("No data available.")
        return

    analysis = NetworkAnalysis()
    result = analysis.run(df)

    soc = result.metrics.get("span_of_control", {})
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Span of Control", f"{soc.get('mean', 0):.1f}")
    with col2:
        st.metric("Max Span", soc.get("max", 0))
    with col3:
        st.metric("Total Supervisors", result.metrics.get("total_supervisors", 0))

    if soc:
        st.subheader("Span of Control Summary")
        st.dataframe(
            pd.DataFrame(list(soc.items()), columns=["Metric", "Value"]),
            use_container_width=True,
        )

    top = result.metrics.get("top_influencers", [])
    if top:
        st.subheader("Top Influencers (Betweenness Centrality)")
        st.dataframe(pd.DataFrame(top), use_container_width=True)
