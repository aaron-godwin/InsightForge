import streamlit as st
from groq import Groq
from load_data import load_data_and_kb
from retriever import InsightRetriever

# ---------------------------------------------------------
# Initialization
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "conversation_summary" not in st.session_state:
    st.session_state.conversation_summary = ""

# Load data + KB once at startup
df, kb = load_data_and_kb()
retriever = InsightRetriever(df, kb)

# ---------------------------------------------------------
# Utility: Create Groq client lazily (AFTER secrets load)
# ---------------------------------------------------------
def get_groq_client():
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is missing from Streamlit secrets.")
    return Groq(api_key=api_key)

# ---------------------------------------------------------
# Utility: Compress large stats dicts
# ---------------------------------------------------------
def compress_stats(stats: dict, max_items: int = 20):
    if not isinstance(stats, dict):
        return stats

    compressed = {}
    for key, value in stats.items():
        if isinstance(value, dict):
            items = list(value.items())[:max_items]
            compressed[key] = {k: compress_stats(v, max_items) for k, v in items}
        elif isinstance(value, list):
            compressed[key] = value[:max_items]
        else:
            compressed[key] = value

    return compressed

# ---------------------------------------------------------
# Utility: Detect analytical intent
# ---------------------------------------------------------
def is_analytical_query(query: str) -> bool:
    q = query.lower()
    keywords = [
        "why", "explain", "trend", "forecast", "projection",
        "anomaly", "anomalies", "over time", "pattern", "patterns",
        "drivers", "root cause", "volatility", "seasonality",
        "shift", "shifts", "performance change", "performance shift",
    ]
    return any(k in q for k in keywords)

# ---------------------------------------------------------
# Conversation summarization
# ---------------------------------------------------------
def summarize_history_if_needed():
    if len(st.session_state.chat_history) > 12:
        old_turns = st.session_state.chat_history[:-4]
        text_to_summarize = ""

        for turn in old_turns:
            text_to_summarize += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

        summary_prompt = f"""
You are InsightForge, an AI business intelligence assistant.

Summarize the following conversation into a concise memory that preserves:
- user goals
- important facts
- key decisions
- relevant context

Do NOT include fluff.

Conversation to summarize:
{text_to_summarize}
"""

        try:
            client = get_groq_client()
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2,
            )
            summary = response.choices[0].message["content"]
            st.session_state.conversation_summary = summary
        except Exception:
            pass

        st.session_state.chat_history = st.session_state.chat_history[-4:]

# ---------------------------------------------------------
# Build unified prompt
# ---------------------------------------------------------
def build_unified_prompt(question: str, stats, analytical_mode: bool) -> str:
    history_text = ""
    for turn in st.session_state.chat_history[-6:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    summary_text = st.session_state.conversation_summary

    if not analytical_mode:
        return f"""
You are InsightForge, an AI business intelligence assistant.

Conversation summary:
{summary_text}

Recent conversation:
{history_text}

User question:
{question}

Relevant structured statistics:
{stats}

Task:
Provide a clear, concise answer based strictly on the statistics above.
"""

    return f"""
You are InsightForge, an AI business intelligence assistant.

Conversation summary:
{summary_text}

Recent conversation:
{history_text}

User question:
{question}

Relevant structured statistics:
{stats}

Your task:
1. Provide a short narrative summary (2–4 sentences).
2. Then provide a structured breakdown with markdown headings.
"""

# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def run_query(question: str) -> str:
    # Step 1 — Retrieve stats
    stats = retriever.retrieve(question)

    # Step 2 — Compress stats
    stats = compress_stats(stats, max_items=20)

    # Step 3 — Decide mode
    analytical = False
    if isinstance(stats, dict):
        stats_type = stats.get("type")
        complex_types = {
            "trend_stats",
            "anomaly_stats",
            "forecast_context",
            "product_region_month_stats",
            "region_consistency",
        }
        if stats_type in complex_types:
            analytical = True

    if is_analytical_query(question):
        analytical = True

    # Step 4 — Summarize history
    summarize_history_if_needed()

    # Step 5 — Build prompt
    prompt = build_unified_prompt(question, stats, analytical_mode=analytical)

    # Step 6 — Call Groq LLM
    try:
        client = get_groq_client()
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are InsightForge, an AI BI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message["content"]

        # Step 7 — Save turn
        st.session_state.chat_history.append(
            {"user": question, "assistant": answer}
        )

        return answer

    except Exception as e:
        return f"Error running Groq query: {e}"