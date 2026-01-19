import streamlit as st
from src.load_data import load_data_and_kb
from src.visualization import InsightVisualizer

st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

st.title("📊 InsightForge — AI‑Powered Business Intelligence")
st.write("Interact with your data, explore visualizations, and generate insights.")

# Load data + KB
df, kb = load_data_and_kb()
viz = InsightVisualizer(df, kb)

# Sidebar navigation
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a view:",
    [
        "Sales Trends",
        "Product Performance",
        "Regional Analysis",
        "Customer Demographics",
    ]
)

# Sales Trends
if page == "Sales Trends":
    st.header("📈 Sales Trends Over Time")
    st.pyplot(viz.plot_sales_trend())
    st.pyplot(viz.plot_monthly_sales())

# Product Performance
elif page == "Product Performance":
    st.header("📦 Product Performance")
    st.pyplot(viz.plot_product_performance())
    st.pyplot(viz.plot_product_region_heatmap())

# Regional Analysis
elif page == "Regional Analysis":
    st.header("🌎 Regional Sales Analysis")
    st.pyplot(viz.plot_region_performance())

# Customer Demographics
elif page == "Customer Demographics":
    st.header("👥 Customer Demographics")
    st.pyplot(viz.plot_age_group_sales())
    st.pyplot(viz.plot_gender_sales())
    st.pyplot(viz.plot_age_gender_matrix())