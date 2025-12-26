import pandas as pd

fact_sales = pd.read_parquet("data/processed/fact_sales.parquet")
cal = pd.read_csv("data/raw/calendar.csv", engine='python')
prices = pd.read_csv("data/raw/sell_prices.csv", engine='python')

cal_min = cal[["date", "wm_yr_wk"]].copy()
cal_min["date"]=pd.to_datetime(cal_min["date"])

daily_sales = pd.merge(fact_sales, cal_min, how='left', on='date')

prices_min = prices[["store_id", "item_id", "wm_yr_wk","sell_price"]].copy()

daily_sales_prices = pd.merge(daily_sales, prices_min, how='left', on=['store_id','item_id', 'wm_yr_wk'])

daily_sales_prices.to_parquet("data/processed/daily_sales_prices.parquet", index=False)


