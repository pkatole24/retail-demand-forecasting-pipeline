import pandas as pd
import numpy as np

df = pd.read_parquet("data/processed/daily_features.parquet")

df["day_of_week"] = df["date"].dt.dayofweek

feature_cols = [
    "lag_7", "lag_28",
    "roll_mean_7", "roll_mean_28",
    "price_lag_1w", "price_change_1w", "price_roll_mean_4w",
    "day_of_week"
]

df_model = df.dropna(subset=feature_cols + ["units_sold"]).copy()

# time-based split
cutoff_date = pd.to_datetime("2015-01-01")

train = df_model[df_model["date"] < cutoff_date]
test = df_model[df_model["date"] >= cutoff_date]

print("Train rows:", train.shape[0])
print("Test rows:", test.shape[0])

# X/y split
X_train = train[feature_cols]
y_train = train["units_sold"]

X_test  = test[feature_cols]
y_test  = test["units_sold"]

train_sample = train.sample(n=1_000_000, random_state=42)

X_train_s = train_sample[feature_cols]
y_train_s = train_sample["units_sold"]

from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(
    n_estimators=200, random_state=42, n_jobs=-1, max_depth=20, min_samples_leaf=5
)

rf.fit(X_train_s, y_train_s)

test_sample = test.sample(n=500_000, random_state=42)
X_test_s = test_sample[feature_cols]
y_test_s = test_sample["units_sold"]

y_pred = rf.predict(X_test_s)

mae = np.mean(np.abs(y_test_s - y_pred))
print("RF MAE:", mae)
print("Baseline MAE (lag_7):", 1.139130520935826)

importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
print(importances)

def wape(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(y_true)

rf_wape = wape(y_test_s, y_pred)
print("RF WAPE:", rf_wape)


