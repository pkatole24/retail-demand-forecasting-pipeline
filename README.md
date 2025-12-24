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
