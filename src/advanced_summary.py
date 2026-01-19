from load_data import load_data_and_kb
import pandas as pd

df, kb = load_data_and_kb()

print("\n================= ADVANCED DATA SUMMARY =================\n")

# ---------------------------------------------------------
# 1. Sales performance by time period
# ---------------------------------------------------------

print("\n--- SALES PERFORMANCE BY TIME PERIOD ---")

# Daily total sales
daily_sales = df.groupby("Date")["Sales"].sum().reset_index()

# Monthly total sales (already in kb["monthly_sales"], but we can enrich)
monthly_sales = kb["monthly_sales"].copy()
monthly_sales["Month"] = monthly_sales["Month"].astype(str)

print("\nDaily sales (first 5 rows):")
print(daily_sales.head())

print("\nMonthly total sales:")
print(monthly_sales)

print("\nOverall sales trend:")
print(f"Total Sales: {df['Sales'].sum():,.0f}")
print(f"Average Daily Sales: {daily_sales['Sales'].mean():,.2f}")
print(f"Average Monthly Sales: {monthly_sales['Sales'].mean():,.2f}")


# ---------------------------------------------------------
# 2. Product and regional analysis
# ---------------------------------------------------------

print("\n--- PRODUCT AND REGIONAL ANALYSIS ---")

product_summary = kb["product_summary"]
region_summary = kb["region_summary"]

print("\nProduct performance summary:")
print(product_summary)

print("\nRegion performance summary:")
print(region_summary)

# Product-region matrix
product_region_sales = df.pivot_table(
    index="Product",
    columns="Region",
    values="Sales",
    aggfunc="sum"
)

print("\nProduct-Region Sales Matrix (Total Sales):")
print(product_region_sales)


# ---------------------------------------------------------
# 3. Customer segmentation by demographics
# ---------------------------------------------------------

print("\n--- CUSTOMER SEGMENTATION BY DEMOGRAPHICS ---")

# Age groups already created in load_data.build_knowledge_base
age_summary = kb["age_summary"]

print("\nAverage Sales by Age Group:")
print(age_summary)

# Gender-based analysis
gender_sales = df.groupby("Customer_Gender")["Sales"].agg(["count", "mean", "sum"]).reset_index()
gender_satisfaction = df.groupby("Customer_Gender")["Customer_Satisfaction"].mean().reset_index()

print("\nSales by Gender:")
print(gender_sales)

print("\nAverage Satisfaction by Gender:")
print(gender_satisfaction)

# Age + Gender segmentation
age_gender_sales = df.groupby(["Age_Group", "Customer_Gender"])["Sales"].mean().reset_index()
print("\nAverage Sales by Age Group and Gender:")
print(age_gender_sales)


# ---------------------------------------------------------
# 4. Statistical measures
# ---------------------------------------------------------

print("\n--- STATISTICAL MEASURES ---")

sales_stats = df["Sales"].agg(["mean", "median", "std", "min", "max"])
satisfaction_stats = df["Customer_Satisfaction"].agg(["mean", "median", "std", "min", "max"])
age_stats = df["Customer_Age"].agg(["mean", "median", "std", "min", "max"])

print("\nSales Statistics:")
print(sales_stats)

print("\nCustomer Satisfaction Statistics:")
print(satisfaction_stats)

print("\nCustomer Age Statistics:")
print(age_stats)