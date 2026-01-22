from load_data import load_data_and_kb
from retriever import InsightRetriever
from prompting import (
    build_insight_prompt,
    build_forecast_prompt,
    build_trend_prompt,
    build_anomaly_prompt,
    build_product_region_month_prompt
)
from groq import Groq
import streamlit as st
import os


def compress_stats(stats: dict, max_items: int = 20):
    """
    Reduce the size of large stats dictionaries so they fit within Groq's token limits.
    Keeps only the first N items in any nested dict or list.
    """
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


# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load data + KB once at startup
df, kb = load_data_and_kb()
retriever = InsightRetriever(df, kb)


# ---------------------------------------------------------
# Chat History Initialization
# ---------------------------------------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


def build_history_text():
    """
    Convert the last 6 turns of chat history into a text block
    that can be injected into the LLM prompt.
    """
    history = st.session_state.chat_history[-6:]  # last 6 turns only
    text = ""

    for turn in history:
        text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    return text


def run_query(question: str) -> str:
    """
    Full RAG pipeline using Groq:
    1. Retrieve relevant statistics
    2. Compress stats to avoid token overflow
    3. Build the correct LLM prompt (insight, trend, anomaly, forecast, etc.)
    4. Add chat history context
    5. Generate an answer using Llama‑3.1‑8B
    6. Save the turn to chat history
    7. Return the insight text
    """

    # Step 1 — Retrieve stats
    stats = retriever.retrieve(question)

    # Step 2 — Compress stats to avoid Groq 413 errors
    stats = compress_stats(stats, max_items=20)

    # Step 3 — Prompt routing
    if isinstance(stats, dict):
        stype = stats.get("type")

        if stype == "forecast_context":
            core_prompt = build_forecast_prompt(question, stats)

        elif stype == "trend_stats":
            core_prompt = build_trend_prompt(question, stats)

        elif stype == "anomaly_stats":
            core_prompt = build_anomaly_prompt(question, stats)

        elif stype == "product_region_month_stats":
            core_prompt = build_product_region_month_prompt(question, stats)

        else:
            core_prompt = build_insight_prompt(question, stats)

    else:
        # Fallback for string messages like "No matching statistics found"
        core_prompt = build_insight_prompt(question, stats)

    # Step 4 — Add chat history to the prompt
    history_text = build_history_text()

    final_prompt = f"""
You are InsightForge, an AI business intelligence assistant.

Conversation so far:
{history_text}

User question:
{question}

Your task:
Use the conversation context AND the structured statistics below to produce a clear, grounded, business-focused insight.

Structured statistics:
{core_prompt}
"""

    # Step 5 — Call Groq LLM
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are InsightForge, an AI BI assistant."},
                {"role": "user", "content": final_prompt}
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content

        # Step 6 — Save turn to chat history
        st.session_state.chat_history.append({
            "user": question,
            "assistant": answer
        })

        return answer

    except Exception as e:
        return f"Error running Groq query: {e}"