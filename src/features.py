import pandas as pd

daily_sales_prices = pd.read_parquet("data/processed/daily_sales_prices.parquet")

daily_sales_prices = daily_sales_prices.sort_values(by=['store_id', 'item_id', 'date'])

# lag features

daily_sales_prices["lag_7"] = (
    daily_sales_prices.groupby(["store_id", "item_id"])["units_sold"].shift(7)
)

daily_sales_prices["lag_28"] = (
    daily_sales_prices.groupby(["store_id", "item_id"])["units_sold"].shift(28)
)

# rolling averages

daily_sales_prices["roll_mean_7"]=(
    daily_sales_prices
    .groupby(["store_id", "item_id"])["units_sold"]
    .shift(1)
    .rolling(window=7)
    .mean()
)

daily_sales_prices["roll_mean_28"]=(
    daily_sales_prices
    .groupby(["store_id", "item_id"])["units_sold"]
    .shift(1)
    .rolling(window=28)
    .mean()
)

# price features

daily_sales_prices["price_lag_1w"] = (
    daily_sales_prices
    .groupby(["store_id", "item_id"])["sell_price"]
    .shift(7)
)

daily_sales_prices["price_change_1w"] = (
    (daily_sales_prices["sell_price"] - daily_sales_prices["price_lag_1w"])
    / daily_sales_prices["price_lag_1w"]
)

daily_sales_prices["price_roll_mean_4w"] = (
    daily_sales_prices
    .groupby(["store_id", "item_id"])["sell_price"]
    .shift(1)
    .rolling(window=28)
    .mean()
)

daily_sales_prices.to_parquet(
    "data/processed/daily_features.parquet",
    index=False
)

