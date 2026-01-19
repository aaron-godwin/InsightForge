def kb_to_text_chunks(kb):
    chunks = []

    for _, row in kb["product_summary"].iterrows():
        chunks.append(
            f"Product {row['Product']} has total sales {row[('Sales','sum')]:.2f}, "
            f"average sales {row[('Sales','mean')]:.2f}, max sale {row[('Sales','max')]:.2f}, "
            f"and average satisfaction {row[('Customer_Satisfaction','mean')]:.2f}."
        )

    for _, row in kb["region_summary"].iterrows():
        chunks.append(
            f"In region {row['Region']}, total sales are {row[('Sales','sum')]:.2f}, "
            f"average sales {row[('Sales','mean')]:.2f}, and average satisfaction "
            f"{row[('Customer_Satisfaction','mean')]:.2f}."
        )

    # You can add similar chunks for monthly_sales, age_summary, etc.
    return chunks