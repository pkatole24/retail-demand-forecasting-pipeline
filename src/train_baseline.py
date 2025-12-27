import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/daily_features.parquet")

df = df.dropna(subset=["lag_7"])

# time-based splitting
cutoff_date = pd.to_datetime("2015-01-01")

df_train = df[df["date"] < cutoff_date]
df_test = df[df["date"] >= cutoff_date]

print("Train rows:", df_train.shape[0])
print("Test rows:", df_test.shape[0])

# baseline model: y_hat = lag_7 and metric: MAE
y_true = df_test["units_sold"]
y_pred = df_test["lag_7"]

mae = np.mean(np.abs(y_true - y_pred))
print("Baseline MAE:", mae)

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

baseline_wape = wape(y_true, y_pred)
print("Baseline WAPE:", baseline_wape)

