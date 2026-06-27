import streamlit as st
import pandas as pd

from wf_analysis.analysis.forecasting import ForecastingAnalysis


def show(df=None):
    st.header("Workforce Planning")

    if df is None:
        st.info("No data available.")
        return

    analysis = ForecastingAnalysis()
    result = analysis.run(df)

    col1, col2 = st.columns(2)
    with col1:
        hires = result.metrics.get("hires_by_year", {})
        if hires:
            st.subheader("Hires by Year")
            st.bar_chart(pd.Series(hires))
    with col2:
        exits = result.metrics.get("exits_by_year", {})
        if exits:
            st.subheader("Exits by Year")
            st.bar_chart(pd.Series(exits))

    for fig in result.plots:
        st.plotly_chart(fig, use_container_width=True)

    trend = result.metrics.get("attrition_trend", {})
    if trend:
        st.subheader("Attrition Trend Data")
        st.dataframe(
            pd.DataFrame(trend).T.round(3),
            use_container_width=True,
        )
