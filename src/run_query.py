from load_data import load_data_and_kb
from retriever import InsightRetriever
from prompting import build_insight_prompt
from groq import Groq
import streamlit as st
import os

# Load Groq API key from Streamlit secrets
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# Load data + KB once at startup
df, kb = load_data_and_kb()
retriever = InsightRetriever(df, kb)

def run_query(question: str) -> str:
    """
    Full RAG pipeline using Groq:
    1. Retrieve relevant statistics
    2. Build an LLM prompt
    3. Generate an answer using Llama‑3.1‑8B
    4. Return the insight text
    """

    # Step 1 — Retrieve stats
    stats = retriever.retrieve(question)

    # Step 2 — Build prompt
    prompt = build_insight_prompt(question, stats)

    # Step 3 — Call Groq LLM
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": "You are InsightForge, an AI BI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        answer = response.choices[0].message.content
        return answer

    except Exception as e:
        return f"Error running Groq query: {e}"