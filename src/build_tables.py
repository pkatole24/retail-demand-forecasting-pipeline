import pandas as pd

# Load raw sales + calendar
df_sales = pd.read_csv("data/raw/sales_train_validation.csv", engine='python')
calendar = pd.read_csv("data/raw/calendar.csv", engine='python')

# Identify d_ columns
columns = df_sales.columns.to_list()

day_cols = [col for col in columns if col.startswith('d_')]
id_cols = [col for col in columns if not col.startswith('d_')]

# Unpivot sales to long format
sales_long = df_sales.melt(id_vars=["item_id", "store_id"],
                           value_vars=day_cols,
                           var_name="d",
                           value_name="units_sold")

# Join calendar to add real date
calendar_min = calendar[["d", "date"]].copy()
calendar_min["date"] = pd.to_datetime(calendar_min["date"])

sales_with_date = pd.merge(sales_long, calendar_min, how="left", on="d")

# Select final columns and enforce dtypes
fact_sales = sales_with_date[["item_id", "store_id", "date", "units_sold"]].copy()
fact_sales["units_sold"]=fact_sales["units_sold"].astype(int)

# Run quality checks
dup_count = fact_sales.duplicated(
    subset=["date", "store_id", "item_id"]
).sum()
print(dup_count)

neg_count = (fact_sales["units_sold"] < 0).sum()
print(neg_count)

print("Null dates:", fact_sales["date"].isna().sum())
print("Null stores:", fact_sales["store_id"].isna().sum())
print("Null items:", fact_sales["item_id"].isna().sum())

# Save to data/processed/fact_sales.parquet
fact_sales.to_parquet("data/processed/fact_sales.parquet", index=False)
