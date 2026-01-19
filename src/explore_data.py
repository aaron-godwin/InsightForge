from load_data import load_data_and_kb

# Load dataset + knowledge base
df, kb = load_data_and_kb()

# Raw data preview
print("\n--- RAW DATA ---")
print(df.head())
print(df.info())

# Product summary
print("\n--- PRODUCT SUMMARY ---")
print(kb["product_summary"])

# Region summary
print("\n--- REGION SUMMARY ---")
print(kb["region_summary"])

# Monthly sales
print("\n--- MONTHLY SALES ---")
print(kb["monthly_sales"])

# Age group summary
print("\n--- AGE GROUP SUMMARY ---")
print(kb["age_summary"])