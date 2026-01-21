import pandas as pd

class InsightRetriever:
    def __init__(self, df, kb):
        self.df = df
        self.kb = kb

    # ---------------------------------------------------------
    # Core helpers
    # ---------------------------------------------------------
    def _ensure_month_column(self):
        """Ensure a Month column exists, derived from Date if needed."""
        if "Month" not in self.df.columns and "Date" in self.df.columns:
            self.df["Date"] = pd.to_datetime(self.df["Date"])
            self.df["Month"] = self.df["Date"].dt.to_period("M").astype(str)

    # ---------------------------------------------------------
    # Product-level statistics
    # ---------------------------------------------------------
    def get_product_stats(self, product):
        subset = self.df[self.df["Product"] == product]

        if subset.empty:
            return f"No data found for product '{product}'."

        return {
            "type": "product_stats",
            "product": product,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "max_sale": subset["Sales"].max(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Region-level statistics
    # ---------------------------------------------------------
    def get_region_stats(self, region):
        subset = self.df[self.df["Region"] == region]

        if subset.empty:
            return f"No data found for region '{region}'."

        return {
            "type": "region_stats",
            "region": region,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Month-level statistics
    # ---------------------------------------------------------
    def get_monthly_stats(self, month):
        self._ensure_month_column()
        subset = self.df[self.df["Month"] == month]

        if subset.empty:
            return f"No data found for month '{month}'."

        return {
            "type": "month_stats",
            "month": str(month),
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Demographic statistics (Customer_Age)
    # ---------------------------------------------------------
    def get_age_stats(self, age_value):
        subset = self.df[self.df["Customer_Age"] == age_value]

        if subset.empty:
            return f"No data found for age '{age_value}'."

        return {
            "type": "age_stats",
            "age": age_value,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Demographic statistics (Customer_Gender)
    # ---------------------------------------------------------
    def get_gender_stats(self, gender_value):
        subset = self.df[self.df["Customer_Gender"] == gender_value]

        if subset.empty:
            return f"No data found for gender '{gender_value}'."

        return {
            "type": "gender_stats",
            "gender": gender_value,
            "total_sales": subset["Sales"].sum(),
            "avg_sales": subset["Sales"].mean(),
            "avg_satisfaction": subset["Customer_Satisfaction"].mean()
        }

    # ---------------------------------------------------------
    # Product × Region × Month analysis
    # ---------------------------------------------------------
    def get_product_region_month_stats(self):
        self._ensure_month_column()
        grouped = (
            self.df
            .groupby(["Product", "Region", "Month"])["Sales"]
            .sum()
            .reset_index()
        )

        result = {}
        for _, row in grouped.iterrows():
            product = row["Product"]
            region = row["Region"]
            month = row["Month"]
            sales = row["Sales"]

            result.setdefault(product, {})
            result[product].setdefault(region, {})
            result[product][region][month] = float(sales)

        return {
            "type": "product_region_month_stats",
            "product_region_month_sales": result
        }

    # ---------------------------------------------------------
    # Trend detection (increasing / decreasing / flat)
    # ---------------------------------------------------------
    def get_trend_stats(self):
        self._ensure_month_column()

        # Aggregate total sales per month
        monthly = (
            self.df
            .groupby("Month")["Sales"]
            .sum()
            .sort_index()
        )

        if len(monthly) < 2:
            return {
                "type": "trend_stats",
                "trend": "insufficient_data",
                "monthly_sales": monthly.to_dict()
            }

        # Simple trend: compare first and last
        first = monthly.iloc[0]
        last = monthly.iloc[-1]

        if last > first * 1.05:
            trend = "increasing"
        elif last < first * 0.95:
            trend = "decreasing"
        else:
            trend = "flat"

        return {
            "type": "trend_stats",
            "trend": trend,
            "monthly_sales": monthly.to_dict()
        }

    # ---------------------------------------------------------
    # Anomaly detection (simple z-score on monthly totals)
    # ---------------------------------------------------------
    def get_anomaly_stats(self):
        self._ensure_month_column()

        monthly = (
            self.df
            .groupby("Month")["Sales"]
            .sum()
            .sort_index()
        )

        if len(monthly) < 3:
            return {
                "type": "anomaly_stats",
                "anomalies": [],
                "monthly_sales": monthly.to_dict()
            }

        series = monthly.astype(float)
        mean = series.mean()
        std = series.std()

        if std == 0:
            anomalies = []
        else:
            z_scores = (series - mean) / std
            anomalies = [
                {"month": idx, "sales": float(val), "z_score": float(z)}
                for idx, val, z in zip(series.index, series.values, z_scores.values)
                if abs(z) >= 2.0
            ]

        return {
            "type": "anomaly_stats",
            "anomalies": anomalies,
            "monthly_sales": series.to_dict()
        }

    # ---------------------------------------------------------
    # Forecasting hook (for Groq-powered forecasting)
    # ---------------------------------------------------------
    def get_forecast_context(self, horizon_months: int = 3):
        """
        Returns structured monthly sales history that the LLM (Groq) can use
        to generate a natural-language forecast.
        """
        self._ensure_month_column()

        monthly = (
            self.df
            .groupby("Month")["Sales"]
            .sum()
            .sort_index()
        )

        return {
            "type": "forecast_context",
            "monthly_sales": monthly.to_dict(),
            "horizon_months": horizon_months
        }

    # ---------------------------------------------------------
    # Generic retrieval for LLM queries
    # ---------------------------------------------------------
    def retrieve(self, query):
        query = query.lower()
        self._ensure_month_column()

        # Product queries
        for product in self.df["Product"].unique():
            if product.lower() in query:
                return self.get_product_stats(product)

        # Region queries
        for region in self.df["Region"].unique():
            if region.lower() in query:
                return self.get_region_stats(region)

        # Age queries (Customer_Age) – specific ages
        if "Customer_Age" in self.df.columns:
            for age in self.df["Customer_Age"].dropna().unique():
                if str(age).lower() in query:
                    return self.get_age_stats(age)

        # Gender queries (Customer_Gender)
        if "Customer_Gender" in self.df.columns:
            for gender in self.df["Customer_Gender"].dropna().unique():
                if str(gender).lower() in query:
                    return self.get_gender_stats(gender)

        # Age summary queries (no specific age mentioned)
        if "age" in query or "ages" in query or "age group" in query:
            if "Customer_Age" in self.df.columns:
                age_stats = {}
                for age in self.df["Customer_Age"].dropna().unique():
                    subset = self.df[self.df["Customer_Age"] == age]
                    age_stats[age] = float(subset["Sales"].sum())
                return {
                    "type": "age_sales_summary",
                    "age_sales_summary": age_stats
                }

        # Region consistency / stability queries
        if (
            "consistent" in query
            or "consistency" in query
            or "month-to-month" in query
            or "month to month" in query
            or ("stable" in query and "region" in query)
            or ("stability" in query and "region" in query)
            or ("region" in query and "over time" in query)
            or ("region" in query and "performance" in query and "over time" in query)
            or ("region" in query and "variance" in query)
            or ("region" in query and "volatility" in query)
        ):
            consistency_stats = {}
            for region in self.df["Region"].unique():
                region_df = self.df[self.df["Region"] == region]
                monthly_totals = (
                    region_df.groupby("Month")["Sales"]
                    .sum()
                    .sort_index()
                    .tolist()
                )
                if len(monthly_totals) > 1:
                    std_dev = pd.Series(monthly_totals).std()
                    consistency_stats[region] = {
                        "monthly_totals": monthly_totals,
                        "std_dev": float(std_dev)
                    }
            return {"type": "region_consistency", "region_consistency": consistency_stats}

        # Product × Region × Month analysis
        if (
            # direct phrasing
            "product-region" in query
            or "product region" in query
            or "product–region" in query
            or "product by region" in query
            or "region by product" in query

            # performance phrasing
            or ("product" in query and "region" in query and "performance" in query)
            or ("product" in query and "region" in query and "over time" in query)
            or ("product" in query and "region" in query and "trend" in query)

            # shift / change phrasing
            or ("shift" in query and "region" in query)
            or ("shifts" in query and "region" in query)
            or ("month-to-month" in query and "region" in query)
            or ("month to month" in query and "region" in query)

            # strongest / weakest phrasing
            or ("strongest" in query and "region" in query)
            or ("weakest" in query and "region" in query)

            # comparison phrasing
            or ("compare" in query and "region" in query and "product" in query)
        ):
            return self.get_product_region_month_stats()


        # Trend detection
        if (
            "trend" in query
            or "increasing" in query
            or "decreasing" in query
            or "seasonal" in query
            or "trajectory" in query
            or "long-term" in query
            or "long term" in query
            or ("sales" in query and "over time" in query)
            or ("historical" in query and "sales" in query)
            or ("sales" in query and "direction" in query)
            or ("sales" in query and "movement" in query)
        ):
            return self.get_trend_stats()

        # Anomaly detection
        if (
            "anomaly" in query
            or "anomalies" in query
            or "outlier" in query
            or "outliers" in query
            or "unusual" in query
            or "unexpected" in query
            or "irregular" in query
            or ("deviate" in query and "month" in query)
            or ("deviates" in query and "month" in query)
            or ("deviation" in query and "month" in query)
            or ("significantly" in query and "month" in query)
            or ("norm" in query and "month" in query)
            or ("stand out" in query and "month" in query)
        ):
            return self.get_anomaly_stats()

        # Forecasting
        if "forecast" in query or "predict" in query or "projection" in query:
            # You can parse horizon from query later if you want
            return self.get_forecast_context(horizon_months=3)

        # Month queries (explicit month mention)
        for month in self.df["Month"].unique():
            if str(month).lower() in query:
                return self.get_monthly_stats(month)

        return "No matching statistics found for your query."