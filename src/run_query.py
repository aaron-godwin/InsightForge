from load_data import load_data_and_kb
from retriever import InsightRetriever
from prompting import build_insight_prompt

df, kb = load_data_and_kb()
retriever = InsightRetriever(df, kb)

query = "How does Widget A perform in the West region?"

stats = retriever.retrieve(query)
prompt = build_insight_prompt(query, stats)

print(prompt)