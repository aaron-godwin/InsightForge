import streamlit as st
from load_data import load_data_and_kb
from visualization import InsightVisualizer
from run_query import run_query  # AI Assistant integration

# force Streamlit to reload latest commit

st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")

# ---------------------------------------------------------
# Initialize session state
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "_trigger_rerun" not in st.session_state:
    st.session_state["_trigger_rerun"] = False

# NEW: required for run_query.py
if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# ---------------------------------------------------------
# ChatGPT-style CSS with avatars
# ---------------------------------------------------------
chat_css = """
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}
.user-bubble {
    background-color: #DCF8C6;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 80%;
    align-self: flex-end;
    display: flex;
    gap: 8px;
}
.assistant-bubble {
    background-color: #F1F0F0;
    padding: 10px 14px;
    border-radius: 12px;
    max-width: 80%;
    align-self: flex-start;
    display: flex;
    gap: 8px;
}
.avatar {
    font-size: 24px;
    line-height: 24px;
}
.bubble-text {
    flex-grow: 1;
}
</style>
"""
st.markdown(chat_css, unsafe_allow_html=True)

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
    ],
)

# Clear conversation button (only on AI Assistant page)
if page == "AI Assistant":
    if st.sidebar.button("🧹 Clear Conversation"):
        st.session_state.chat_history = []
        st.session_state["_trigger_rerun"] = True

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

    # Chat history display
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)

    for turn in st.session_state.chat_history:
        # USER BUBBLE
        st.markdown(
            f"""
            <div class="user-bubble">
                <div class="avatar">👤</div>
                <div class="bubble-text"><b>You:</b> {turn["user"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ASSISTANT BUBBLE
        st.markdown(
            f"""
            <div class="assistant-bubble">
                <div class="avatar">🤖</div>
                <div class="bubble-text"><b>Assistant:</b> {turn["assistant"]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # RAW STATS EXPANDER
        if "stats" in turn and turn["stats"] is not None:
            with st.expander("Raw Stats Used"):
                st.json(turn["stats"])

    st.markdown("</div>", unsafe_allow_html=True)

    # User input
    user_question = st.text_area(
        "Your question:",
        placeholder="e.g., Why did sales drop in Q3 in the West region?",
        height=120,
    )

    # Suggested questions
    st.subheader("Suggested questions")
    suggestions = [
        "Which region is performing the best this year?",
        "Are there any anomalies in monthly sales?",
        "What is the forecast for the next quarter?",
        "Which product is gaining momentum over time?",
        "How do customer age groups differ in revenue contribution?",
    ]

    cols = st.columns(2)
    for i, s in enumerate(suggestions):
        if cols[i % 2].button(s):
            with st.spinner("Analyzing with RAG engine..."):
                run_query(s)
            st.session_state["_trigger_rerun"] = True

    # Run analysis button
    if st.button("Run Analysis"):
        if not user_question.strip():
            st.warning("Please enter a question before running analysis.")
        else:
            with st.spinner("Analyzing with RAG engine..."):
                run_query(user_question)
            st.session_state["_trigger_rerun"] = True

# ---------------------------------------------------------
# SAFE RERUN HANDLER — MUST BE AT TOP LEVEL
# ---------------------------------------------------------
if st.session_state.get("_trigger_rerun", False):
    st.session_state["_trigger_rerun"] = False
    st.rerun()