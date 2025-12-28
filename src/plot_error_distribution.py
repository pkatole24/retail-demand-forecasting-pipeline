import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

df = pd.read_parquet("data/processed/daily_features.parquet")
rf = joblib.load("models/rf_v1.joblib")

df["date"] = pd.to_datetime(df["date"])
df["day_of_week"] = df["date"].dt.dayofweek

feature_cols = [
    "lag_7", "lag_28",
    "roll_mean_7", "roll_mean_28",
    "price_lag_1w", "price_change_1w", "price_roll_mean_4w",
    "day_of_week"
]

df = df.dropna(subset=feature_cols + ["units_sold"]).copy()

cutoff = pd.to_datetime("2015-01-01")
test = df[df["date"] >= cutoff].copy()

test_sample = test.sample(n=500_000, random_state=42)

test_sample["pred_baseline"] = test_sample["lag_7"]
test_sample["pred_rf"] = rf.predict(test_sample[feature_cols])

# Absolute errors
test_sample["err_baseline"] = np.abs(
    test_sample["units_sold"] - test_sample["pred_baseline"]
)
test_sample["err_rf"] = np.abs(
    test_sample["units_sold"] - test_sample["pred_rf"]
)

# Plot
plt.figure(figsize=(7,5))
plt.boxplot(
    [test_sample["err_baseline"], test_sample["err_rf"]],
    labels=["Baseline (lag_7)", "Random Forest"],
    showfliers=False
)
plt.ylabel("Absolute Error (Units)")
plt.title("Absolute Error Distribution: Baseline vs RandomForest")
plt.tight_layout()
plt.show()