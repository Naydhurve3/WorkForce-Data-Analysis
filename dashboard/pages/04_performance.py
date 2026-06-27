import streamlit as st
import pandas as pd

from wf_analysis.analysis.performance import PerformanceAnalysis


def show(df=None):
    st.header("Performance Analytics")

    if df is None:
        st.info("No data available.")
        return

    analysis = PerformanceAnalysis()
    result = analysis.run(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Rating", f"{result.metrics.get('avg_rating', 0):.2f}")
    with col2:
        st.metric("PIP Count", result.metrics.get("pip_count", 0))
    with col3:
        st.metric("PIP Rate", f"{result.metrics.get('pip_rate', 0):.1%}")

    for fig in result.plots:
        st.plotly_chart(fig, use_container_width=True)

    score_dist = result.metrics.get("score_distribution", {})
    if score_dist:
        st.subheader("Score Distribution")
        st.dataframe(
            pd.Series(score_dist, name="Count").to_frame(),
            use_container_width=True,
        )
