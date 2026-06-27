import streamlit as st


def kpi_card(
    label: str,
    value: str,
    delta: str | None = None,
    help_text: str | None = None,
    color: str = "#2E86AB",
):
    st.markdown(
        f"""
        <div style="
            background: white;
            border-radius: 8px;
            padding: 16px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.12);
            border-left: 4px solid {color};
            margin-bottom: 8px;
        ">
            <div style="color: #666; font-size: 0.85rem; margin-bottom: 4px;">
                {label}
            </div>
            <div style="font-size: 1.8rem; font-weight: 700; color: #1a1a2e;">
                {value}
            </div>
            {f'<div style="font-size: 0.9rem; color: {"#2ecc71" if delta and not delta.startswith("-") else "#e74c3c"}; margin-top: 2px;">{delta}</div>' if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )
    if help_text:
        st.caption(help_text)
