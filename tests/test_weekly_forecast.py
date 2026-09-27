import numpy as np
import pandas as pd

from src.weekly_forecast import build_weekly_examples, select_portfolio


def make_daily(start: str, end: str, item_values: dict[str, int]) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    frames = []
    for item_id, units in item_values.items():
        frames.append(pd.DataFrame({
            "store_id": "CA_1",
            "item_id": item_id,
            "dept_id": "FOODS_1",
            "cat_id": "FOODS",
            "cohort": "high_volume",
            "date": dates,
            "first_sale_date": dates[0],
            "units_sold": units,
            "sell_price": 2.0,
        }))
    return pd.concat(frames, ignore_index=True)


def test_origin_features_stay_with_the_item_and_exclude_future_sales():
    daily = make_daily("2014-01-01", "2014-04-30", {"A": 1, "B": 10})
    origin = pd.Timestamp("2014-03-02")
    before = build_weekly_examples(daily)
    a_before = before.loc[before["origin"].eq(origin) & before["item_id"].eq("A")].iloc[0]
    b_before = before.loc[before["origin"].eq(origin) & before["item_id"].eq("B")].iloc[0]
    assert (a_before["sales_last_7"], a_before["sales_last_56"], a_before["actual"]) == (7, 56, 7)
    assert (b_before["sales_last_7"], b_before["sales_last_56"], b_before["actual"]) == (70, 560, 70)

    daily.loc[daily["item_id"].eq("A") & daily["date"].eq(origin + pd.Timedelta(days=3)), "units_sold"] = 101
    after = build_weekly_examples(daily)
    a_after = after.loc[after["origin"].eq(origin) & after["item_id"].eq("A")].iloc[0]
    assert a_after["sales_last_7"] == a_before["sales_last_7"]
    assert a_after["sales_last_56"] == a_before["sales_last_56"]
    assert a_after["actual"] == a_before["actual"] + 100


def test_target_weeks_do_not_cross_split_boundaries():
    weekly = build_weekly_examples(make_daily("2014-10-01", "2016-04-24", {"A": 1}))
    train = weekly.loc[weekly["split"].eq("train")]
    validation = weekly.loc[weekly["split"].eq("validation")]
    test = weekly.loc[weekly["split"].eq("test")]
    assert not train.empty and not validation.empty and not test.empty
    assert train["target_end"].lt("2015-01-01").all()
    assert validation["target_start"].ge("2015-01-01").all()
    assert validation["target_end"].lt("2016-01-01").all()
    assert test["target_start"].ge("2016-01-01").all()
    assert weekly["split"].ne("boundary_gap").all()


def test_portfolio_selection_ignores_future_sales():
    dates = pd.date_range("2013-12-31", "2015-01-01", freq="D")
    days = [f"d_{index}" for index in range(1, len(dates) + 1)]
    calendar = pd.DataFrame({"d": days, "date": dates})
    values = np.zeros((201, len(days)), dtype="int32")
    values[:, 0] = 1
    values[:100, 1:366] = 3
    values[100:, 1:366:12] = 1
    sales = pd.DataFrame(values, columns=days)
    sales.insert(0, "store_id", "CA_1")
    sales.insert(1, "item_id", [f"H{i:03d}" for i in range(100)] + [f"S{i:03d}" for i in range(101)])
    sales.insert(2, "dept_id", "FOODS_1")
    sales.insert(3, "cat_id", "FOODS")

    before = select_portfolio(sales, calendar)
    sales.loc[sales["item_id"].eq("S100"), days[-1]] = 1_000_000
    after = select_portfolio(sales, calendar)
    assert before[["store_id", "item_id", "cohort"]].equals(after[["store_id", "item_id", "cohort"]])
