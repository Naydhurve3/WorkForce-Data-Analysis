import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def display_plotly_chart(fig: go.Figure, key: str = "chart", height: int = 400):
    if fig is None:
        st.info("No chart data available.")
        return
    st.plotly_chart(fig, use_container_width=True, key=key)


def display_matplotlib_chart(fig: plt.Figure, caption: str = ""):
    if fig is None:
        st.info("No chart data available.")
        return
    st.pyplot(fig)
    if caption:
        st.caption(caption)


def display_metric_row(metrics: dict, cols: int = 4):
    items = list(metrics.items())
    row_size = (len(items) + cols - 1) // cols
    for i in range(0, len(items), cols):
        row = items[i : i + cols]
        cols_ui = st.columns(len(row))
        for col, (label, value) in zip(cols_ui, row):
            with col:
                if isinstance(value, float):
                    st.metric(label, f"{value:.2f}")
                else:
                    st.metric(label, str(value))
