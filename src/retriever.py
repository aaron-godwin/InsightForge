import pandas as pd


class InsightRetriever:
    def __init__(self, df: pd.DataFrame, kb):
        self.df = df
        self.kb = kb

        # Ensure Month exists for all time-based logic
        self._ensure_month_column()

        # -------------------------------------------------
        # Precomputed aggregates (for fast, consistent stats)
        # -------------------------------------------------
        self.region_totals = (
            self.df.groupby("Region")["Sales"].sum().sort_values(ascending=False).to_dict()
        )

        self.product_totals = (
            self.df.groupby("Product")["Sales"].sum().sort_values(ascending=False).to_dict()
        )

        self.monthly_sales = (
            self.df.groupby("Month")["Sales"].sum().sort_index().to_dict()
        )

        self.product_region_month = (
            self.df.groupby(["Product", "Region", "Month"])["Sales"]
            .sum()
            .reset_index()
        )

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
            return {
                "type": "product_stats",
                "message": f"No data found for product '{product}'.",
            }

        return {
            "type": "product_stats",
            "product": product,
            "total_sales": float(subset["Sales"].sum()),
            "avg_sales": float(subset["Sales"].mean()),
            "max_sale": float(subset["Sales"].max()),
            "avg_satisfaction": float(subset["Customer_Satisfaction"].mean())
            if "Customer_Satisfaction" in subset.columns
            else None,
        }

    # ---------------------------------------------------------
    # Region-level statistics
    # ---------------------------------------------------------
    def get_region_stats(self, region):
        subset = self.df[self.df["Region"] == region]

        if subset.empty:
            return {
                "type": "region_stats",
                "message": f"No data found for region '{region}'.",
            }

        return {
            "type": "region_stats",
            "region": region,
            "total_sales": float(subset["Sales"].sum()),
            "avg_sales": float(subset["Sales"].mean()),
            "avg_satisfaction": float(subset["Customer_Satisfaction"].mean())
            if "Customer_Satisfaction" in subset.columns
            else None,
        }

    # ---------------------------------------------------------
    # Month-level statistics
    # ---------------------------------------------------------
    def get_monthly_stats(self, month):
        subset = self.df[self.df["Month"] == month]

        if subset.empty:
            return {
                "type": "month_stats",
                "message": f"No data found for month '{month}'.",
            }

        return {
            "type": "month_stats",
            "month": str(month),
            "total_sales": float(subset["Sales"].sum()),
            "avg_sales": float(subset["Sales"].mean()),
            "avg_satisfaction": float(subset["Customer_Satisfaction"].mean())
            if "Customer_Satisfaction" in subset.columns
            else None,
        }

    # ---------------------------------------------------------
    # Demographic statistics (Customer_Age)
    # ---------------------------------------------------------
    def get_age_stats(self, age_value):
        subset = self.df[self.df["Customer_Age"] == age_value]

        if subset.empty:
            return {
                "type": "age_stats",
                "message": f"No data found for age '{age_value}'.",
            }

        return {
            "type": "age_stats",
            "age": age_value,
            "total_sales": float(subset["Sales"].sum()),
            "avg_sales": float(subset["Sales"].mean()),
            "avg_satisfaction": float(subset["Customer_Satisfaction"].mean())
            if "Customer_Satisfaction" in subset.columns
            else None,
        }

    # ---------------------------------------------------------
    # Demographic statistics (Customer_Gender)
    # ---------------------------------------------------------
    def get_gender_stats(self, gender_value):
        subset = self.df[self.df["Customer_Gender"] == gender_value]

        if subset.empty:
            return {
                "type": "gender_stats",
                "message": f"No data found for gender '{gender_value}'.",
            }

        return {
            "type": "gender_stats",
            "gender": gender_value,
            "total_sales": float(subset["Sales"].sum()),
            "avg_sales": float(subset["Sales"].mean()),
            "avg_satisfaction": float(subset["Customer_Satisfaction"].mean())
            if "Customer_Satisfaction" in subset.columns
            else None,
        }

    # ---------------------------------------------------------
    # Product × Region × Month analysis
    # ---------------------------------------------------------
    def get_product_region_month_stats(self):
        grouped = (
            self.df.groupby(["Product", "Region", "Month"])["Sales"]
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
            "product_region_month_sales": result,
        }

    # ---------------------------------------------------------
    # Trend detection (increasing / decreasing / flat)
    # ---------------------------------------------------------
    def get_trend_stats(self):
        monthly = (
            self.df.groupby("Month")["Sales"]
            .sum()
            .sort_index()
        )

        if len(monthly) < 2:
            return {
                "type": "trend_stats",
                "trend": "insufficient_data",
                "monthly_sales": monthly.to_dict(),
            }

        first = float(monthly.iloc[0])
        last = float(monthly.iloc[-1])

        if last > first * 1.05:
            trend = "increasing"
        elif last < first * 0.95:
            trend = "decreasing"
        else:
            trend = "flat"

        return {
            "type": "trend_stats",
            "trend": trend,
            "monthly_sales": {k: float(v) for k, v in monthly.to_dict().items()},
        }

    # ---------------------------------------------------------
    # Anomaly detection (simple z-score on monthly totals)
    # ---------------------------------------------------------
    def get_anomaly_stats(self):
        monthly = (
            self.df.groupby("Month")["Sales"]
            .sum()
            .sort_index()
        )

        if len(monthly) < 3:
            return {
                "type": "anomaly_stats",
                "anomalies": [],
                "monthly_sales": {k: float(v) for k, v in monthly.to_dict().items()},
            }

        series = monthly.astype(float)
        mean = float(series.mean())
        std = float(series.std())

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
            "monthly_sales": {k: float(v) for k, v in series.to_dict().items()},
        }

    # ---------------------------------------------------------
    # Forecasting hook (for Groq-powered forecasting)
    # ---------------------------------------------------------
    def get_forecast_context(self, horizon_months: int = 3):
        """
        Returns structured monthly sales history that the LLM can use
        to generate a natural-language forecast.
        """
        monthly = (
            self.df.groupby("Month")["Sales"]
            .sum()
            .reset_index()
        )

        monthly["Month"] = pd.PeriodIndex(monthly["Month"], freq="M").astype(str)
        monthly = monthly.sort_values("Month")

        series = monthly.set_index("Month")["Sales"].astype(float)

        return {
            "type": "forecast_context",
            "monthly_sales": series.to_dict(),
            "horizon_months": horizon_months,
            "meta": {
                "num_months": len(series),
            },
        }

    # ---------------------------------------------------------
    # Region performance (ranking regions by total sales)
    # ---------------------------------------------------------
    def get_region_performance(self):
        ranked = sorted(
            self.region_totals.items(), key=lambda x: x[1], reverse=True
        )
        top_region = ranked[0][0] if ranked else None

        return {
            "type": "region_performance",
            "region_totals": {k: float(v) for k, v in self.region_totals.items()},
            "ranked": [(r, float(v)) for r, v in ranked],
            "top_region": top_region,
        }

    # ---------------------------------------------------------
    # Product performance (ranking products by total sales)
    # ---------------------------------------------------------
    def get_product_performance(self):
        ranked = sorted(
            self.product_totals.items(), key=lambda x: x[1], reverse=True
        )
        top_product = ranked[0][0] if ranked else None

        return {
            "type": "product_performance",
            "product_totals": {k: float(v) for k, v in self.product_totals.items()},
            "ranked": [(p, float(v)) for p, v in ranked],
            "top_product": top_product,
        }

    # ---------------------------------------------------------
    # Region consistency / volatility
    # ---------------------------------------------------------
    def get_region_consistency(self):
        region_volatility = (
            self.df.groupby("Region")["Sales"].std().sort_values().to_dict()
        )
        volatility = {k: float(v) for k, v in region_volatility.items()}

        if not volatility:
            return {
                "type": "region_consistency",
                "volatility": {},
                "most_consistent": None,
                "most_volatile": None,
            }

        most_consistent = min(volatility, key=volatility.get)
        most_volatile = max(volatility, key=volatility.get)

        return {
            "type": "region_consistency",
            "volatility": volatility,
            "most_consistent": most_consistent,
            "most_volatile": most_volatile,
        }

    # ---------------------------------------------------------
    # Generic retrieval for LLM queries (new, clean routing)
    # ---------------------------------------------------------
    def retrieve(self, query: str):
        q = query.lower().strip()

        # 1. Region performance (recommended questions)
        if "region" in q and (
            "best" in q
            or "top" in q
            or "strongest" in q
            or "performing" in q
            or "leader" in q
        ):
            return self.get_region_performance()

        # 2. Product performance
        if "product" in q and (
            "best" in q
            or "top" in q
            or "strongest" in q
            or "performing" in q
            or "leader" in q
        ):
            return self.get_product_performance()

        # 3. Trend questions
        if (
            "trend" in q
            or "over time" in q
            or "how has" in q
            or "trajectory" in q
            or "increasing" in q
            or "decreasing" in q
            or ("sales" in q and "history" in q)
        ):
            return self.get_trend_stats()

        # 4. Anomaly detection
        if (
            "anomaly" in q
            or "anomalies" in q
            or "outlier" in q
            or "outliers" in q
            or "unusual" in q
            or "unexpected" in q
            or "spike" in q
            or "drop" in q
        ):
            return self.get_anomaly_stats()

        # 5. Forecasting
        if (
            "forecast" in q
            or "predict" in q
            or "projection" in q
            or "project" in q
            or "next month" in q
            or "next quarter" in q
            or ("future" in q and "sales" in q)
            or ("expected" in q and "sales" in q)
            or ("outlook" in q and "sales" in q)
        ):
            return self.get_forecast_context(horizon_months=3)

        # 6. Product × Region × Month analysis
        if (
            "product-region" in q
            or "product region" in q
            or "product–region" in q
            or "product by region" in q
            or "region by product" in q
            or ("product" in q and "region" in q and "performance" in q)
            or ("product" in q and "region" in q and "over time" in q)
            or ("product" in q and "region" in q and "trend" in q)
            or ("shift" in q and "region" in q)
            or ("shifts" in q and "region" in q)
            or ("month-to-month" in q and "region" in q)
            or ("month to month" in q and "region" in q)
            or ("strongest" in q and "region" in q)
            or ("weakest" in q and "region" in q)
            or ("compare" in q and "region" in q and "product" in q)
        ):
            return self.get_product_region_month_stats()

        # 7. Region consistency / stability
        if (
            "consistent" in q
            or "consistency" in q
            or "month-to-month" in q
            or "month to month" in q
            or ("stable" in q and "region" in q)
            or ("stability" in q and "region" in q)
            or ("region" in q and "over time" in q)
            or ("region" in q and "variance" in q)
            or ("region" in q and "volatility" in q)
        ):
            return self.get_region_consistency()

        # 8. Age summary queries (no specific age mentioned)
        if "age" in q or "ages" in q or "age group" in q:
            if "Customer_Age" in self.df.columns:
                age_stats = {}
                for age in self.df["Customer_Age"].dropna().unique():
                    subset = self.df[self.df["Customer_Age"] == age]
                    age_stats[age] = float(subset["Sales"].sum())
                return {
                    "type": "age_sales_summary",
                    "age_sales_summary": age_stats,
                }

        # 9. Product queries (specific product mention)
        for product in self.df["Product"].unique():
            if product.lower() in q:
                return self.get_product_stats(product)

        # 10. Region queries (specific region mention)
        for region in self.df["Region"].unique():
            if region.lower() in q:
                return self.get_region_stats(region)

        # 11. Age queries (specific age)
        if "Customer_Age" in self.df.columns:
            for age in self.df["Customer_Age"].dropna().unique():
                if str(age).lower() in q:
                    return self.get_age_stats(age)

        # 12. Gender queries
        if "Customer_Gender" in self.df.columns:
            for gender in self.df["Customer_Gender"].dropna().unique():
                if str(gender).lower() in q:
                    return self.get_gender_stats(gender)

        # 13. Month queries (explicit month mention)
        for month in self.df["Month"].unique():
            m_str = str(month).lower()
            if m_str in q:
                return self.get_monthly_stats(month)

        # 14. Fallback
        return {
            "type": "no_stats",
            "message": "No matching statistics found for your query.",
        }