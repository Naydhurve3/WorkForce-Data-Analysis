import streamlit as st
import pandas as pd

from wf_analysis.analysis.attrition import AttritionAnalysis


def show(df=None):
    st.header("Attrition Explorer")

    if df is None:
        st.info("No data available. Run the pipeline first.")
        return

    analysis = AttritionAnalysis()
    result = analysis.run(df)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Attrition Rate", f"{result.metrics.get('attrition_rate', 0):.1%}")
    with col2:
        st.metric("Attrited", result.metrics.get("attrition_count", 0))
    with col3:
        st.metric("Total", result.metrics.get("total_count", 0))

    for fig in result.plots:
        st.plotly_chart(fig, use_container_width=True)

    if "by_department" in result.metrics:
        st.subheader("Attrition by Department")
        dept_df = pd.DataFrame(result.metrics["by_department"]).T
        st.dataframe(dept_df, use_container_width=True)

    if "termination_type" in result.metrics:
        st.subheader("Termination Types")
        st.dataframe(
            pd.Series(result.metrics["termination_type"]).to_frame("Count"),
            use_container_width=True,
        )
