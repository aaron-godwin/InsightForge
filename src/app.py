import streamlit as st
from load_data import load_data_and_kb
from visualization import InsightVisualizer
from run_query import run_query  # AI Assistant integration

st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

# ---------------------------------------------------------
# Initialize chat history
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------
# Title + Intro
# ---------------------------------------------------------
st.title("📊 InsightForge — AI‑Powered Business Intelligence")
st.write(
    "Explore your data visually or ask natural‑language questions powered by your full RAG system."
)

# ---------------------------------------------------------
# Load data + knowledge base
# ---------------------------------------------------------
df, kb = load_data_and_kb()
viz = InsightVisualizer(df, kb)

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Choose a view:",
    [
        "Sales Trends",
        "Product Performance",
        "Regional Analysis",
        "Customer Demographics",
        "AI Assistant",
    ]
)

# Add Clear Conversation button in sidebar
if page == "AI Assistant":
    if st.sidebar.button("🧹 Clear Conversation"):
        st.session_state.chat_history = []
        st.experimental_rerun()

# ---------------------------------------------------------
# Sales Trends
# ---------------------------------------------------------
if page == "Sales Trends":
    st.header("📈 Sales Trends Over Time")
    st.pyplot(viz.plot_sales_trend())
    st.pyplot(viz.plot_monthly_sales())

# ---------------------------------------------------------
# Product Performance
# ---------------------------------------------------------
elif page == "Product Performance":
    st.header("📦 Product Performance")
    st.pyplot(viz.plot_product_performance())
    st.pyplot(viz.plot_product_region_heatmap())

# ---------------------------------------------------------
# Regional Analysis
# ---------------------------------------------------------
elif page == "Regional Analysis":
    st.header("🌎 Regional Sales Analysis")
    st.pyplot(viz.plot_region_performance())

# ---------------------------------------------------------
# Customer Demographics
# ---------------------------------------------------------
elif page == "Customer Demographics":
    st.header("👥 Customer Demographics")
    st.pyplot(viz.plot_age_group_sales())
    st.pyplot(viz.plot_gender_sales())
    st.pyplot(viz.plot_age_gender_matrix())

# ---------------------------------------------------------
# AI Assistant (RAG-powered)
# ---------------------------------------------------------
elif page == "AI Assistant":
    st.header("🤖 AI‑Powered BI Assistant")

    st.write(
        "Ask any question about your data or documents. "
        "InsightForge will use your full RAG pipeline to retrieve relevant context and generate insights."
    )

    # Display chat history
    for turn in st.session_state.chat_history:
        st.markdown(f"**You:** {turn['user']}")
        st.markdown(f"**Assistant:** {turn['assistant']}")
        st.markdown("---")

    # User input (NO KEY to avoid StreamlitAPIException)
    user_question = st.text_area(
        "Your question:",
        placeholder="e.g., Why did sales drop in Q3 in the West region?",
        height=120
    )

    if st.button("Run Analysis"):
        if not user_question.strip():
            st.warning("Please enter a question before running analysis.")
        else:
            with st.spinner("Analyzing with RAG engine..."):
                try:
                    answer = run_query(user_question)

                    # Add to history
                    st.session_state.chat_history.append({
                        "user": user_question,
                        "assistant": answer
                    })

                    # Rerun to refresh chat display and clear input
                    st.experimental_rerun()

                except Exception as e:
                    st.error("An error occurred while running the AI assistant.")
                    st.exception(e)