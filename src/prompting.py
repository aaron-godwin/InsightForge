def build_interpretation_prompt(query):
    return f"""
You are InsightForge, an AI Business Intelligence Assistant.

Your task is to interpret the user's query and identify:
- The metrics they are asking about
- The entities involved (product, region, time period, demographic)
- The type of analysis required (trend, comparison, summary, etc.)

User query:
"{query}"

Respond with a JSON object containing:
- "metrics"
- "entities"
- "analysis_type"
"""


def build_insight_prompt(query, retrieved_stats):
    return f"""
You are InsightForge, an AI Business Intelligence Assistant.

The user asked:
"{query}"

Relevant statistics retrieved from the dataset:
{retrieved_stats}

Using ONLY the statistics above:
- Provide a clear, accurate insight
- Explain what the numbers mean
- Highlight trends or anomalies
- Avoid assumptions or invented data
- Keep the explanation concise and business-focused
"""


def build_refinement_prompt(insight_text):
    return f"""
Refine the following business insight to be:
- Clear
- Concise
- Professional
- Actionable

Insight:
{insight_text}

Return the improved version.
"""