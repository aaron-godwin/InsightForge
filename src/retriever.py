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
    # Retrieve demographic statistics (Customer_Age)
    # ---------------------------------------------------------
    def get_age_stats(self, age_value):
        subset = self.df[self.df["Customer_Age"] == age_value]

        if subset.empty:
            return f"No data found for age '{age_value}'."

        return {
            "age": age_value,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Retrieve demographic statistics (Customer_Gender)
    # ---------------------------------------------------------
    def get_gender_stats(self, gender_value):
        subset = self.df[self.df["Customer_Gender"] == gender_value]

        if subset.empty:
            return f"No data found for gender '{gender_value}'."

        return {
            "gender": gender_value,
            "total_sales": subset["Sales"].sum(),
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

        # Age queries (Customer_Age)
        if "age" in query or "ages" in query or "age group" in query:
            if "Customer_Age" in self.df.columns:
                age_stats = {}
                for age in self.df["Customer_Age"].dropna().unique():
                    subset = self.df[self.df["Customer_Age"] == age]
                    age_stats[age] = subset["Sales"].sum()
                return {"age_sales_summary": age_stats}

        # Gender queries (Customer_Gender)
        if "Customer_Gender" in self.df.columns:
            for gender in self.df["Customer_Gender"].dropna().unique():
                if str(gender).lower() in query:
                    return self.get_gender_stats(gender)

        # Month queries
        for month in self.df["Month"].unique():
            if str(month).lower() in query:
                return self.get_monthly_stats(month)

        # Region consistency queries
        if "consistent" in query or "consistency" in query or "month-to-month" in query:
            if "Region" in self.df.columns and "Month" in self.df.columns:
                consistency_stats = {}

                for region in self.df["Region"].unique():
                    region_df = self.df[self.df["Region"] == region]

                    # Compute monthly totals
                    monthly_totals = (
                        region_df.groupby("Month")["Sales"]
                        .sum()
                        .sort_index()
                        .tolist()
                    )

                    if len(monthly_totals) > 1:
                        # Standard deviation as a consistency measure
                        std_dev = pd.Series(monthly_totals).std()
                        consistency_stats[region] = {
                            "monthly_totals": monthly_totals,
                            "std_dev": float(std_dev)
                        }

                return {"region_consistency": consistency_stats}

        return "No matching statistics found for your query."