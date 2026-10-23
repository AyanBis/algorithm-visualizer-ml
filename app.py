"""Algorithm Visualizer ML - Main Streamlit Application."""

import streamlit as st

from ui.components import render_header, render_sidebar, render_page


def main():
    st.set_page_config(
        page_title="Algorithm Visualizer ML",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for better appearance
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.5rem;
    }
    div[data-testid="stExpander"] details {
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
    }
    .stAlert {
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

    render_header()
    page = render_sidebar()
    render_page(page)


if __name__ == "__main__":
    main()