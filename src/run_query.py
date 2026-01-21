from load_data import load_data_and_kb
from retriever import InsightRetriever
from prompting import build_insight_prompt
import openai
import streamlit as st
import os

# Load API key from Streamlit secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# Load data + KB once at startup
df, kb = load_data_and_kb()
retriever = InsightRetriever(df, kb)

def run_query(question: str) -> str:
    """
    Full RAG pipeline:
    1. Retrieve relevant statistics
    2. Build an LLM prompt
    3. Generate an answer
    4. Return the insight text
    """

    # Step 1 — Retrieve stats
    stats = retriever.retrieve(question)

    # Step 2 — Build prompt
    prompt = build_insight_prompt(question, stats)

    # Step 3 — Call LLM
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are InsightForge, an AI BI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
        )

        answer = response["choices"][0]["message"]["content"]
        return answer

    except Exception as e:
        return f"Error running query: {e}"