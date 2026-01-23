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

# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load data + KB once at startup
df, kb = load_data_and_kb()
retriever = InsightRetriever(df, kb)


# ---------------------------------------------------------
# Utility: Compress large stats dicts
# ---------------------------------------------------------
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


# ---------------------------------------------------------
# Utility: Detect analytical intent from query
# ---------------------------------------------------------
def is_analytical_query(query: str) -> bool:
    q = query.lower()
    keywords = [
        "why",
        "explain",
        "trend",
        "forecast",
        "projection",
        "anomaly",
        "anomalies",
        "over time",
        "pattern",
        "patterns",
        "drivers",
        "root cause",
        "volatility",
        "seasonality",
        "shift",
        "shifts",
        "performance change",
        "performance shift",
    ]
    return any(k in q for k in keywords)


# ---------------------------------------------------------
# Conversation summarization
# ---------------------------------------------------------
def summarize_history_if_needed():
    """
    When chat history gets long, summarize older turns into a compact memory
    and keep only the most recent turns.
    """
    # If more than 12 turns, summarize the oldest ones
    if len(st.session_state.chat_history) > 12:
        old_turns = st.session_state.chat_history[:-4]  # keep last 4 turns
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

Do NOT include fluff. Focus on what matters for future questions.

Conversation to summarize:
{text_to_summarize}
"""

        try:
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[{"role": "user", "content": summary_prompt}],
                temperature=0.2,
            )
            summary = response.choices[0].message["content"]
            st.session_state.conversation_summary = summary
        except Exception:
            # If summarization fails, keep existing summary and history
            pass

        # Keep only the last 4 turns to control prompt size
        st.session_state.chat_history = st.session_state.chat_history[-4:]


# ---------------------------------------------------------
# Build unified prompt (simple vs analytical)
# ---------------------------------------------------------
def build_unified_prompt(question: str, stats, analytical_mode: bool) -> str:
    """
    Unified prompt builder that:
    - uses conversation summary
    - uses recent chat history
    - switches between simple and analytical modes
    """
    # Build recent history text
    history_text = ""
    for turn in st.session_state.chat_history[-6:]:
        history_text += f"User: {turn['user']}\nAssistant: {turn['assistant']}\n\n"

    summary_text = st.session_state.conversation_summary

    # Simple mode: direct answer, minimal structure
    if not analytical_mode:
        return f"""
You are InsightForge, an AI business intelligence assistant.

Conversation summary (for context):
{summary_text}

Recent conversation:
{history_text}

User question:
{question}

Relevant structured statistics:
{stats}

Task:
Provide a clear, concise answer to the user's question based strictly on the statistics above.
Do NOT invent numbers. If something is not in the data, say so briefly.
"""

    # Analytical mode: hybrid narrative + structured breakdown
    return f"""
You are InsightForge, an AI business intelligence assistant.

Conversation summary (for context):
{summary_text}

Recent conversation:
{history_text}

User question:
{question}

Relevant structured statistics:
{stats}

Your task:
1. First, provide a short narrative summary (2–4 sentences) explaining what is happening in the data.
2. Then, provide a structured breakdown with clear sections using markdown headings and bullet points.

Use the following structure when possible:

- Narrative Summary (2–4 sentences)
- Trend
- Anomalies
- Product × Region × Month (if relevant)
- Forecast (if relevant)
- Recommendations

Guidelines:
- Ground every statement in the provided statistics.
- Do NOT invent specific numbers that are not present in the stats.
- Use uncertainty language when discussing forecasts or projections.
- Write like a BI analyst briefing an executive.
"""


# ---------------------------------------------------------
# Main entry point
# ---------------------------------------------------------
def run_query(question: str) -> str:
    """
    Full RAG pipeline using Groq:
    1. Retrieve relevant statistics
    2. Compress stats to avoid token overflow
    3. Decide simple vs analytical mode
    4. Summarize history if needed
    5. Build unified prompt with history + summary
    6. Generate an answer using Llama‑3.1‑8B
    7. Save the turn to chat history
    8. Return the insight text
    """
    # Step 1 — Retrieve stats
    stats = retriever.retrieve(question)

    # Step 2 — Compress stats to avoid Groq 413 errors
    stats = compress_stats(stats, max_items=20)

    # Step 3 — Decide mode (simple vs analytical)
    analytical = False
    stats_type = None

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

    # Also trigger analytical mode based on query intent
    if is_analytical_query(question):
        analytical = True

    # Step 4 — Summarize history if needed
    summarize_history_if_needed()

    # Step 5 — Build unified prompt
    prompt = build_unified_prompt(question, stats, analytical_mode=analytical)

    # Step 6 — Call Groq LLM
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are InsightForge, an AI BI assistant."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message["content"]

        # Step 7 — Save turn to chat history
        st.session_state.chat_history.append(
            {
                "user": question,
                "assistant": answer,
            }
        )

        # Step 8 — Return answer
        return answer

    except Exception as e:
        return f"Error running Groq query: {e}"

print("run_query was called")