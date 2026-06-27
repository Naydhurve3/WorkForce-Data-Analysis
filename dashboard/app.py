import streamlit as st
import pandas as pd

from components.kpi_card import kpi_card
from components.filter_sidebar import render_filters, apply_filters
from components.chart_wrapper import display_plotly_chart, display_metric_row
from utils import load_processed_data

st.set_page_config(
    page_title="WorkForce Analytics v2.0",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

with open("assets/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def main():
    st.sidebar.title("📊 WorkForce Analytics")
    st.sidebar.markdown("---")

    df = load_processed_data()
    if df is not None:
        filters = render_filters(df)
        df_filtered = apply_filters(df, filters)
    else:
        df_filtered = None

    pages = {
        "Executive Overview": "01_overview",
        "Attrition Explorer": "02_attrition",
        "Diversity & Inclusion": "03_diversity",
        "Performance Analytics": "04_performance",
        "NLP Insights": "05_nlp_insights",
        "Org Network": "06_org_network",
        "Workforce Planning": "07_workforce_planning",
    }

    page_key = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
    page_module = pages[page_key]

    st.sidebar.markdown("---")
    if df is not None:
        st.sidebar.metric("Filtered Records", f"{len(df_filtered):,}" if df_filtered is not None else "N/A")
    st.sidebar.caption("WorkForce Analytics v2.0")

    try:
        page = __import__(f"pages.{page_module}", fromlist=["show"])
        page.show(df_filtered if df_filtered is not None else df)
    except ImportError:
        st.title(f"📊 {page_key}")
        st.info(f"🚧 **{page_key}** page is under construction.")
    except Exception as e:
        st.title(f"📊 {page_key}")
        st.error(f"Error loading page: {e}")


if __name__ == "__main__":
    main()
