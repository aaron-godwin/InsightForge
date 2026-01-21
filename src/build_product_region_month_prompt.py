def build_product_region_month_prompt(question: str, stats: dict) -> str:
    data = stats.get("product_region_month_sales", {})

    return f"""
You are InsightForge, an AI business intelligence analyst.

Your task is to analyze product performance across regions and months using ONLY the structured data provided below. 
Do NOT invent or assume any additional data.

---

### Product × Region × Month Sales (Structured)
{data}

---

### Instructions
1. **Summarize key patterns**
   - Identify which products perform best in each region.
   - Highlight regions where certain products consistently outperform others.
   - Note any month-to-month shifts in performance.

2. **Identify strengths and weaknesses**
   - Which product–region combinations are strongest?
   - Which combinations underperform or show volatility?

3. **Detect meaningful patterns**
   - Seasonal effects
   - Regional preferences
   - Product momentum or decline

4. **Provide BI-style insights**
   - Explain the “why” behind observed patterns.
   - Offer actionable recommendations for sales, marketing, or inventory.

---

### Critical Rules
- Do NOT fabricate missing months or regions.
- Do NOT assume seasonality unless clearly visible.
- Base all reasoning strictly on the provided data.

---

User Question: "{question}"
"""