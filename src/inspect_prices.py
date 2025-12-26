import pandas as pd

prices = pd.read_csv("data/raw/sell_prices.csv", engine='python')

print(prices.shape)

cols = prices.columns.to_list()

print(cols)
print(prices.head)