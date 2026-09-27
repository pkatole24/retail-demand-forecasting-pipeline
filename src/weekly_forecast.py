"""Forecast at Sunday close using observations through Sunday; score the following Monday-Sunday total."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import lightgbm
import numpy as np
import pandas as pd
import statsforecast
from lightgbm import LGBMRegressor
from statsforecast.models import CrostonOptimized


SEED = 42
MODELS = ("last_week", "four_week_average", "croston_optimized", "lightgbm")
CATEGORICAL_FEATURES = ("store_id", "item_id", "dept_id", "cat_id")
MODEL_FEATURES = (
    "sales_last_7",
    "sales_last_28",
    "sales_last_56",
    "nonzero_days_last_28",
    "price_last_observed",
    "price_change_28d",
    "price_unavailable",
    "week_of_year",
    "month",
    "days_since_first_sale",
    *CATEGORICAL_FEATURES,
)


def load_calendar(path: Path) -> pd.DataFrame:
    calendar = pd.read_csv(path, parse_dates=["date"])
    if calendar["d"].duplicated().any() or calendar["date"].duplicated().any():
        raise ValueError("Calendar must have one row per day and M5 day identifier")
    if calendar["date"].isna().any() or calendar["wm_yr_wk"].isna().any():
        raise ValueError("Calendar has missing dates or week identifiers")
    return calendar[["d", "date", "wm_yr_wk"]]


def select_portfolio(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    days_2014 = calendar.loc[calendar["date"].dt.year.eq(2014), "d"].tolist()
    before_2015 = calendar.loc[calendar["date"].lt("2015-01-01"), "d"].tolist()
    if not set(days_2014).issubset(sales.columns):
        raise ValueError("Sales file does not contain all 2014 dates")

    values_2014 = sales[days_2014]
    portfolio = sales[["store_id", "item_id", "dept_id", "cat_id"]].copy()
    portfolio["volume_2014"] = values_2014.sum(axis=1)
    portfolio["nonzero_rate_2014"] = values_2014.gt(0).mean(axis=1)

    first_positive = sales[before_2015].gt(0)
    first_day = first_positive.idxmax(axis=1)
    day_dates = calendar.set_index("d")["date"]
    portfolio["first_sale_date"] = first_day.map(day_dates)
    eligible = portfolio.loc[
        portfolio["volume_2014"].ge(10)
        & portfolio["first_sale_date"].le("2013-12-31")
    ].copy()
    ordered = eligible.sort_values(
        ["volume_2014", "store_id", "item_id"], ascending=[False, True, True]
    )
    high = ordered.head(100).copy()
    sparse = ordered.loc[
        ~ordered.index.isin(high.index)
        & ordered["nonzero_rate_2014"].between(0.01, 0.10)
    ].head(100).copy()
    if len(high) != 100 or len(sparse) != 100:
        raise ValueError("Expected 100 high-volume and 100 sparse eligible series")
    high["cohort"] = "high_volume"
    sparse["cohort"] = "sparse"
    selected = pd.concat([high, sparse], ignore_index=True)
    if selected.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Portfolio selection produced duplicate series")
    return selected


def load_selected_prices(path: Path, selected: pd.DataFrame) -> pd.DataFrame:
    keys = selected[["store_id", "item_id"]]
    chunks = []
    for chunk in pd.read_csv(
        path,
        usecols=["store_id", "item_id", "wm_yr_wk", "sell_price"],
        chunksize=500_000,
    ):
        chunks.append(chunk.merge(keys, on=["store_id", "item_id"], how="inner"))
    prices = pd.concat(chunks, ignore_index=True)
    if prices.duplicated(["store_id", "item_id", "wm_yr_wk"]).any():
        raise ValueError("Prices contain duplicate store-item-week keys")
    if prices["sell_price"].dropna().le(0).any():
        raise ValueError("Prices must be positive when present")
    return prices


def make_daily_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    calendar = load_calendar(data_dir / "calendar.csv")
    day_columns = calendar.loc[calendar["date"].le("2016-04-24"), "d"].tolist()
    header = pd.read_csv(data_dir / "sales_train_validation.csv", nrows=0).columns
    day_columns = [column for column in day_columns if column in header]
    if not day_columns:
        raise ValueError("No M5 sales day columns found")
    id_columns = ["store_id", "item_id", "dept_id", "cat_id"]
    sales = pd.read_csv(
        data_dir / "sales_train_validation.csv",
        usecols=id_columns + day_columns,
        dtype={column: "int32" for column in day_columns},
    )
    if sales.duplicated(["store_id", "item_id"]).any():
        raise ValueError("Sales contain duplicate store-item series")
    if sales[day_columns].isna().any().any() or sales[day_columns].lt(0).any().any():
        raise ValueError("Sales contain missing or negative units")

    selected = select_portfolio(sales, calendar)
    chosen = sales.merge(
        selected[["store_id", "item_id"]], on=["store_id", "item_id"],
        how="inner", validate="one_to_one",
    )
    daily = chosen.melt(
        id_vars=id_columns, value_vars=day_columns, var_name="d", value_name="units_sold"
    ).merge(calendar, on="d", how="left", validate="many_to_one")
    if daily["date"].isna().any() or len(daily) != len(selected) * len(day_columns):
        raise ValueError("Sales to calendar join changed the expected daily grain")
    prices = load_selected_prices(data_dir / "sell_prices.csv", selected)
    daily = daily.merge(
        prices, on=["store_id", "item_id", "wm_yr_wk"], how="left", validate="many_to_one"
    )
    daily = daily.merge(
        selected[["store_id", "item_id", "cohort", "first_sale_date"]],
        on=["store_id", "item_id"], how="left", validate="many_to_one",
    )
    return daily.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True), selected


def build_weekly_examples(daily: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for _, group in daily.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date").reset_index(drop=True)
        sales = group["units_sold"].astype("float64")
        price = group["sell_price"].ffill()
        sunday = group["date"].dt.dayofweek.eq(6)
        examples = group.loc[sunday, [
            "store_id", "item_id", "dept_id", "cat_id", "cohort", "date", "first_sale_date"
        ]].copy()
        examples = examples.rename(columns={"date": "origin"})
        for window in (7, 28, 56):
            examples[f"sales_last_{window}"] = sales.rolling(window, min_periods=window).sum().loc[sunday].to_numpy()
        examples["nonzero_days_last_28"] = (
            sales.gt(0).rolling(28, min_periods=28).sum().loc[sunday].to_numpy()
        )
        examples["price_last_observed"] = price.loc[sunday].to_numpy()
        previous_price = price.shift(28)
        examples["price_change_28d"] = (
            (price / previous_price - 1).replace([np.inf, -np.inf], np.nan).loc[sunday].to_numpy()
        )
        examples["price_unavailable"] = price.isna().loc[sunday].astype("int8").to_numpy()
        future_total = sum(sales.shift(-day) for day in range(1, 8))
        examples["actual"] = future_total.loc[sunday].to_numpy()
        examples["days_since_first_sale"] = (
            examples["origin"] - examples["first_sale_date"]
        ).dt.days
        frames.append(examples)

    weekly = pd.concat(frames, ignore_index=True)
    weekly["target_start"] = weekly["origin"] + pd.Timedelta(days=1)
    weekly["target_end"] = weekly["origin"] + pd.Timedelta(days=7)
    weekly["week_of_year"] = weekly["target_start"].dt.isocalendar().week.astype("int16")
    weekly["month"] = weekly["target_start"].dt.month.astype("int8")
    weekly = weekly.loc[
        weekly["days_since_first_sale"].ge(56)
        & weekly["actual"].notna()
        & weekly["sales_last_56"].notna()
    ].copy()
    weekly["split"] = np.select(
        [
            weekly["target_end"].lt("2015-01-01"),
            weekly["target_start"].ge("2015-01-01") & weekly["target_end"].lt("2016-01-01"),
            weekly["target_start"].ge("2016-01-01"),
        ],
        ["train", "validation", "test"], default="boundary_gap",
    )
    weekly = weekly.loc[weekly["split"].ne("boundary_gap")].copy()
    for column in CATEGORICAL_FEATURES:
        weekly[column] = weekly[column].astype("category")
    if weekly.duplicated(["origin", "store_id", "item_id"]).any():
        raise ValueError("Weekly examples are not unique by origin and series")
    return weekly.sort_values(["origin", "store_id", "item_id"]).reset_index(drop=True)


def croston_forecasts(daily: pd.DataFrame, examples: pd.DataFrame) -> np.ndarray:
    histories = {}
    for key, group in daily.groupby(["store_id", "item_id"], sort=False):
        group = group.sort_values("date")
        histories[key] = (
            group["date"].to_numpy(dtype="datetime64[ns]"),
            group["units_sold"].to_numpy(dtype="float64"),
            np.datetime64(group["first_sale_date"].iloc[0]),
        )
    model = CrostonOptimized()
    forecasts = []
    for row in examples.itertuples(index=False):
        dates, sales, first_sale = histories[(row.store_id, row.item_id)]
        start = np.searchsorted(dates, first_sale, side="left")
        end = np.searchsorted(dates, np.datetime64(row.origin), side="right")
        forecast = model.forecast(y=sales[start:end], h=7)["mean"]
        forecasts.append(float(np.maximum(forecast, 0).sum()))
    return np.asarray(forecasts)


def metric_rows(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with_all = pd.concat([predictions, predictions.assign(cohort="all")], ignore_index=True)
    for (split, cohort), group in with_all.groupby(["split", "cohort"], observed=True):
        for model in MODELS:
            error = group[model].to_numpy() - group["actual"].to_numpy()
            actual_sum = group["actual"].sum()
            rows.append({
                "split": split,
                "cohort": cohort,
                "model": model,
                "n_forecasts": len(group),
                "actual_units": float(actual_sum),
                "mae": float(np.abs(error).mean()),
                "wape": float(np.abs(error).sum() / actual_sum) if actual_sum > 0 else np.nan,
                "bias_units": float(error.mean()),
            })
    return pd.DataFrame(rows).sort_values(["split", "cohort", "model"]).reset_index(drop=True)


def train_and_score(weekly: pd.DataFrame, daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, LGBMRegressor, str]:
    train = weekly.loc[weekly["split"].eq("train")]
    validation = weekly.loc[weekly["split"].eq("validation")]
    test = weekly.loc[weekly["split"].eq("test")]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Training, validation, and test must each contain eligible weeks")
    settings = dict(
        n_estimators=300, learning_rate=0.05, num_leaves=31,
        min_child_samples=40, random_state=SEED, n_jobs=-1, verbosity=-1,
    )
    model = LGBMRegressor(**settings)
    model.fit(train[list(MODEL_FEATURES)], train["actual"])

    frames = []
    for part in (validation, test):
        scored = part[[
            "origin", "target_start", "target_end", "store_id", "item_id", "cohort", "split", "actual"
        ]].copy()
        scored["last_week"] = part["sales_last_7"].to_numpy()
        scored["four_week_average"] = part["sales_last_28"].to_numpy() / 4
        scored["croston_optimized"] = croston_forecasts(daily, part)
        if part is validation:
            scored["lightgbm"] = np.maximum(model.predict(part[list(MODEL_FEATURES)]), 0)
        frames.append(scored)

    validation_metrics = metric_rows(frames[0])
    winner = validation_metrics.loc[
        validation_metrics["cohort"].eq("all")
    ].sort_values(["wape", "model"]).iloc[0]["model"]
    model = LGBMRegressor(**settings)
    development = pd.concat([train, validation], ignore_index=True)
    model.fit(development[list(MODEL_FEATURES)], development["actual"])
    frames[1]["lightgbm"] = np.maximum(model.predict(test[list(MODEL_FEATURES)]), 0)
    predictions = pd.concat(frames, ignore_index=True)
    if predictions[list(MODELS)].isna().any().any():
        raise ValueError("A model produced missing forecasts")
    return predictions, metric_rows(predictions), model, winner


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(data_dir: Path, output_dir: Path) -> None:
    input_paths = [data_dir / name for name in (
        "sales_train_validation.csv", "calendar.csv", "sell_prices.csv"
    )]
    for path in input_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "run.json"
    manifest_path.unlink(missing_ok=True)

    daily, selected = make_daily_data(data_dir)
    weekly = build_weekly_examples(daily)
    predictions, metrics, model, winner = train_and_score(weekly, daily)
    selected.to_csv(output_dir / "selected_series.csv", index=False)
    predictions.to_parquet(output_dir / "predictions.parquet", index=False)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    joblib.dump(model, output_dir / "lightgbm.joblib")
    manifest = {
        "forecast_contract": "Sunday close to following Monday-Sunday total units",
        "portfolio_rule": "2014-only: top 100 volume and 100 active series with 1-10% nonzero days",
        "n_series": len(selected),
        "n_train_examples": int(weekly["split"].eq("train").sum()),
        "n_validation_forecasts": int(predictions["split"].eq("validation").sum()),
        "n_test_forecasts": int(predictions["split"].eq("test").sum()),
        "selected_on_validation": winner,
        "evaluation_note": "The 2016 period was inspected before sparse-cohort refinement; its scores are exploratory.",
        "input_sha256": {path.name: sha256(path) for path in input_paths},
        "versions": {
            "python": platform.python_version(), "pandas": pd.__version__,
            "lightgbm": lightgbm.__version__, "statsforecast": statsforecast.__version__,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(metrics.loc[metrics["cohort"].eq("all")].to_string(index=False))
    print(f"Validation-selected model: {winner}")
    print(f"Saved results to {output_dir.resolve()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/weekly"))
    args = parser.parse_args()
    run(args.data_dir, args.output_dir)


if __name__ == "__main__":
    main()
