import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

sns.set(style="whitegrid")


class InsightVisualizer:

    def __init__(self, df: pd.DataFrame, kb: dict):
        self.df = df
        self.kb = kb

    # ---------------------------------------------------------
    # Sales Trends
    # ---------------------------------------------------------
    def plot_sales_trend(self):
        daily_sales = self.df.groupby("Date")["Sales"].sum().reset_index()

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=daily_sales, x="Date", y="Sales", marker="o", ax=ax)
        ax.set_title("Daily Sales Trend")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    def plot_monthly_sales(self):
        monthly_sales = self.kb["monthly_sales"]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=monthly_sales, x="Month", y="Sales", palette="Blues_d", ax=ax)
        ax.set_title("Monthly Sales Totals")
        plt.xticks(rotation=45)
        plt.tight_layout()
        return fig

    # ---------------------------------------------------------
    # Product Performance
    # ---------------------------------------------------------
    def plot_product_performance(self):
        product_summary = self.kb["product_summary"]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=product_summary,
            x="Product",
            y="Sales_sum",
            palette="Greens_d",
            ax=ax
        )
        ax.set_title("Product Performance (Total Sales)")
        plt.tight_layout()
        return fig

    # ---------------------------------------------------------
    # Regional Performance
    # ---------------------------------------------------------
    def plot_region_performance(self):
        region_summary = self.kb["region_summary"]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=region_summary,
            x="Region",
            y="Sales_sum",
            palette="Purples_d",
            ax=ax
        )
        ax.set_title("Regional Sales Performance")
        plt.tight_layout()
        return fig

    def plot_product_region_heatmap(self):
        pivot = self.df.pivot_table(
            index="Product",
            columns="Region",
            values="Sales",
            aggfunc="sum"
        )

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd", ax=ax)
        ax.set_title("Product × Region Sales Heatmap")
        plt.tight_layout()
        return fig

    # ---------------------------------------------------------
    # Demographics
    # ---------------------------------------------------------
    def plot_age_group_sales(self):
        age_summary = self.kb["age_summary"]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            data=age_summary,
            x="Customer_Age",
            y="Average_Sales",
            palette="Oranges_d",
            ax=ax
        )
        ax.set_title("Average Sales by Customer Age")
        plt.tight_layout()
        return fig

    def plot_gender_sales(self):
        gender_summary = self.kb["gender_summary"]

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(
            data=gender_summary,
            x="Customer_Gender",
            y="Total_Sales",
            palette="coolwarm",
            ax=ax
        )
        ax.set_title("Total Sales by Gender")
        plt.tight_layout()
        return fig

    def plot_age_gender_matrix(self):
        pivot = self.kb["age_gender_matrix"]

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="BuPu", ax=ax)
        ax.set_title("Average Sales by Age × Gender")
        plt.tight_layout()
        return fig