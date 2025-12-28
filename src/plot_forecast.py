import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

df = pd.read_parquet("data/processed/daily_features.parquet")
rf = joblib.load("models/rf_v1.joblib")

df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek

feature_cols = [
    "lag_7", "lag_28",
    "roll_mean_7", "roll_mean_28",
    "price_lag_1w", "price_change_1w", "price_roll_mean_4w",
    "day_of_week"
]

df = df.dropna(subset=feature_cols + ["units_sold"]).copy()
df["date"] = pd.to_datetime(df["date"])

cutoff = pd.to_datetime("2015-01-01")
test = df[df["date"] >= cutoff].copy()

series_scores = (
    test.groupby(["store_id", "item_id"])["units_sold"]
        .sum()
        .sort_values(ascending=False)
)

store_id, item_id = series_scores.index[0]

# 90-day window in test period
start = pd.to_datetime("2015-03-01")
end   = pd.to_datetime("2015-05-31")

s = test[(test["store_id"] == store_id) & (test["item_id"] == item_id)].copy()
s = s[(s["date"] >= start) & (s["date"] <= end)].sort_values("date")

# Predictions
s["pred_baseline"] = s["lag_7"]
s["pred_rf"] = rf.predict(s[feature_cols])

# Plot
plt.figure(figsize=(12, 5))
plt.plot(s["date"], s["units_sold"])
plt.plot(s["date"], s["pred_baseline"])
plt.plot(s["date"], s["pred_rf"])
plt.title(f"Actual vs Baseline vs RandomForest | store={store_id}, item={item_id}")
plt.xlabel("Date")
plt.ylabel("Units sold")
plt.legend(["Actual", "Baseline (lag_7)", "RandomForest"])
plt.tight_layout()
plt.show()