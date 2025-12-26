# M5 Ops Forecasting Pipeline

## Project Goal
Build an end-to-end operations analytics pipeline that converts raw retail sales data into clean, structured tables and forecasting-ready features.

## Data Source
M5 Forecasting – Accuracy dataset (Kaggle), containing daily item-level sales, calendar metadata, and weekly prices.

## How to Run
1. Create and activate the conda environment.
2. Install dependencies from requirements.txt.
3. Run:
   python src/build_tables.py

## Outputs
- data/processed/fact_sales.parquet  
  Clean fact table at (date, store_id, item_id) grain.

## Data Storage Decisions

### Why Parquet
Processed tables are stored in Parquet format instead of CSV because Parquet:
- Preserves column data types (dates, integers) reliably
- Loads significantly faster for large tables
- Is the industry standard for analytics and ML pipelines

This matters because the daily analytic table contains tens of millions of rows and is read repeatedly during feature engineering and modeling.

### Missing Prices (NaNs)
After aligning weekly prices to daily sales, some rows have missing `sell_price` values.

This is expected and not a join error. It occurs because:
- Items not priced in certain stores or weeks
- Items introduced after the start of the sales history
- Incomplete price coverage in early periods

A left join is intentionally used to preserve the full sales history. Missing prices are handled explicitly during feature engineering.
