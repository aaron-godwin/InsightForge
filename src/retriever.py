import pandas as pd

class InsightRetriever:
    def __init__(self, df, kb):
        self.df = df
        self.kb = kb

    # ---------------------------------------------------------
    # Retrieve product-level statistics
    # ---------------------------------------------------------
    def get_product_stats(self, product):
        subset = self.df[self.df["Product"] == product]

        if subset.empty:
            return f"No data found for product '{product}'."

        return {
            "product": product,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "max_sale": subset["Sales"].max(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Retrieve region-level statistics
    # ---------------------------------------------------------
    def get_region_stats(self, region):
        subset = self.df[self.df["Region"] == region]

        if subset.empty:
            return f"No data found for region '{region}'."

        return {
            "region": region,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Retrieve time-based statistics
    # ---------------------------------------------------------
    def get_monthly_stats(self, month):
        subset = self.df[self.df["Month"] == month]

        if subset.empty:
            return f"No data found for month '{month}'."

        return {
            "month": str(month),
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Retrieve demographic statistics
    # ---------------------------------------------------------
    def get_age_group_stats(self, age_group):
        subset = self.df[self.df["Age_Group"] == age_group]

        if subset.empty:
            return f"No data found for age group '{age_group}'."

        return {
            "age_group": age_group,
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Generic retrieval for LLM queries
    # ---------------------------------------------------------
    def retrieve(self, query):
        query = query.lower()

        # Product queries
        for product in self.df["Product"].unique():
            if product.lower() in query:
                return self.get_product_stats(product)

        # Region queries
        for region in self.df["Region"].unique():
            if region.lower() in query:
                return self.get_region_stats(region)

        # Age group queries
        for group in self.df["Age_Group"].dropna().unique():
            if str(group).lower() in query:
                return self.get_age_group_stats(group)

        # Month queries
        for month in self.df["Month"].unique():
            if str(month).lower() in query:
                return self.get_monthly_stats(month)

        return "No matching statistics found for your query."