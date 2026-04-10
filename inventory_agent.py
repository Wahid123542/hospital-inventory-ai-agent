import pandas as pd

# Load data
df = pd.read_csv("hospital_inventory_data.csv")

# Add days remaining
df["days_remaining"] = df["current_stock"] / df["daily_usage"]

# Define stock status
def stock_status(row):
    if row["current_stock"] < row["reorder_threshold"]:
        return "LOW"
    elif row["days_remaining"] < row["supplier_lead_time_days"]:
        return "RISK"
    else:
        return "OK"

df["stock_status"] = df.apply(stock_status, axis=1)

# Define action
def reorder_recommendation(row):
    if row["stock_status"] == "LOW":
        return "Reorder Immediately"
    elif row["stock_status"] == "RISK":
        return "Reorder Soon"
    else:
        return "Stock OK"

df["action"] = df.apply(reorder_recommendation, axis=1)

# Show important results
print("\n=== INVENTORY STATUS ===")
print(df[["medication_name", "department", "current_stock", "days_remaining", "stock_status", "action"]].head(15))

# Show summary
print("\n=== SUMMARY ===")
print(df["stock_status"].value_counts())

# Show critical items
print("\n=== LOW STOCK ITEMS ===")
print(df[df["stock_status"] == "LOW"][["medication_name", "current_stock", "reorder_threshold"]].head())

print("\n=== AT RISK ITEMS ===")
print(df[df["stock_status"] == "RISK"][["medication_name", "days_remaining", "supplier_lead_time_days"]].head())