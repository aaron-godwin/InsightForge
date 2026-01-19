import os
import pandas as pd


def load_data_and_kb():
    """
    Loads the sales dataset and builds a structured knowledge base (KB)
    containing product, region, monthly, and demographic summaries.
    """

    # ---------------------------------------------------------
    # Load dataset using a relative path (works locally + Streamlit Cloud)
    # ---------------------------------------------------------
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "sales_data.csv")

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # ---------------------------------------------------------
    # Preprocess dataset
    # ---------------------------------------------------------
    df["Date"] = pd.to_datetime(df["Date"])
    df["Month"] = df["Date"].dt.to_period("M").astype(str)

    kb = {}

    # ---------------------------------------------------------
    # Product Summary
    # ---------------------------------------------------------
    product_summary = df.groupby("Product").agg({
        "Sales": ["sum", "mean", "max"],
        "Customer_Satisfaction": "mean"
    }).reset_index()

    product_summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in product_summary.columns
    ]

    kb["product_summary"] = product_summary

    # ---------------------------------------------------------
    # Region Summary
    # ---------------------------------------------------------
    region_summary = df.groupby("Region").agg({
        "Sales": ["sum", "mean", "max"],
        "Customer_Satisfaction": "mean"
    }).reset_index()

    region_summary.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0]
        for col in region_summary.columns
    ]

    kb["region_summary"] = region_summary

    # ---------------------------------------------------------
    # Monthly Sales Summary
    # ---------------------------------------------------------
    monthly_sales = df.groupby("Month")["Sales"].sum().reset_index()
    monthly_sales.columns = ["Month", "Sales"]
    kb["monthly_sales"] = monthly_sales

    # ---------------------------------------------------------
    # Age Summary
    # ---------------------------------------------------------
    age_summary = df.groupby("Customer_Age")["Sales"].mean().reset_index()
    age_summary.columns = ["Customer_Age", "Average_Sales"]
    kb["age_summary"] = age_summary

    # ---------------------------------------------------------
    # Gender Summary
    # ---------------------------------------------------------
    gender_summary = df.groupby("Customer_Gender")["Sales"].sum().reset_index()
    gender_summary.columns = ["Customer_Gender", "Total_Sales"]
    kb["gender_summary"] = gender_summary

    # ---------------------------------------------------------
    # Age × Gender Matrix
    # ---------------------------------------------------------
    age_gender_matrix = df.pivot_table(
        index="Customer_Age",
        columns="Customer_Gender",
        values="Sales",
        aggfunc="mean"
    )
    kb["age_gender_matrix"] = age_gender_matrix

    return df, kb