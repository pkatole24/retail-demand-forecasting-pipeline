import pandas as pd

sales = pd.read_csv("data/raw/sales_train_validation.csv", engine='python')

print(sales.shape)

all_columns = sales.columns.to_list()

print(all_columns[:10])
print(all_columns[-10:])
