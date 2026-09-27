from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from openai import OpenAIError

from src.analyst_summary import api_configured, summarize_results
from src.weekly_forecast import MODELS


ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "data" / "weekly"
LABELS = {
    "last_week": "Last week",
    "four_week_average": "Four-week average",
    "croston_optimized": "Croston optimized",
    "lightgbm": "LightGBM",
}


@st.cache_data
def load_results(directory: str, run_mtime: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = Path(directory)
    predictions = pd.read_parquet(path / "predictions.parquet")
    metrics = pd.read_csv(path / "metrics.csv")
    manifest = json.loads((path / "run.json").read_text(encoding="utf-8"))
    return predictions, metrics, manifest


def main() -> None:
    st.set_page_config(page_title="M5 Forecast Workbench", layout="wide")
    st.title("M5 weekly demand forecast workbench")
    st.caption("Sunday-close forecasts for the next seven days. Historical daily model scores use a different target and are not directly comparable.")
    run_path = ARTIFACTS / "run.json"
    if not run_path.is_file():
        st.info("No completed run found. From the repository root, run `python -m src.weekly_forecast` first.")
        return
    predictions, metrics, manifest = load_results(str(ARTIFACTS), run_path.stat().st_mtime)
    st.write(
        f"**{manifest['n_series']} series** · "
        f"**{manifest['n_validation_forecasts']:,} validation forecasts** · "
        f"**{manifest['n_test_forecasts']:,} test forecasts**"
    )
    st.write(f"Selected on 2015 validation WAPE: **{LABELS[manifest['selected_on_validation']]}**")
    st.info(manifest["evaluation_note"])

    left, right = st.columns([1, 2])
    with left:
        split = st.selectbox("Evaluation period", ["test", "validation"], format_func=str.title)
        cohort = st.selectbox("Demand group", ["all", "high_volume", "sparse"], format_func=lambda value: value.replace("_", " ").title())
        model = st.selectbox("Model to inspect", list(MODELS), index=list(MODELS).index(manifest["selected_on_validation"]), format_func=lambda value: LABELS[value])
    subset_metrics = metrics.loc[metrics["split"].eq(split) & metrics["cohort"].eq(cohort)].copy()
    subset_metrics["model"] = subset_metrics["model"].map(LABELS)
    with right:
        st.subheader("Like-for-like model comparison")
        st.dataframe(
            subset_metrics[["model", "n_forecasts", "mae", "wape", "bias_units"]].rename(columns={
                "model": "Model", "n_forecasts": "Forecasts", "mae": "MAE (units)",
                "wape": "WAPE", "bias_units": "Bias (predicted − actual)",
            }),
            hide_index=True, use_container_width=True,
        )

    subset = predictions.loc[predictions["split"].eq(split)].copy()
    if cohort != "all":
        subset = subset.loc[subset["cohort"].eq(cohort)]
    store = st.selectbox("Store", sorted(subset["store_id"].unique()))
    item_choices = sorted(subset.loc[subset["store_id"].eq(store), "item_id"].unique())
    item = st.selectbox("Item", item_choices)
    series = subset.loc[subset["store_id"].eq(store) & subset["item_id"].eq(item)].sort_values("target_start")
    st.subheader(f"Seven-day demand · {store} / {item}")
    chart = series.set_index("target_start")[["actual", model]].rename(columns={"actual": "Actual", model: LABELS[model]})
    st.line_chart(chart)

    subset["absolute_error"] = (subset[model] - subset["actual"]).abs()
    st.subheader("Largest misses in this group")
    st.dataframe(
        subset.nlargest(15, "absolute_error")[[
            "target_start", "store_id", "item_id", "actual", model, "absolute_error"
        ]].rename(columns={model: LABELS[model]}),
        hide_index=True, use_container_width=True,
    )
    st.caption("These are offline forecast diagnostics for a possible replenishment use case; no inventory cost or business savings were measured.")

    st.subheader("Optional analyst summary")
    if not api_configured(ROOT):
        st.info("Add OPENAI_API_KEY and OPENAI_MODEL to the ignored .env file to enable an on-demand summary.")
    elif st.button("Explain these results"):
        payload = {
            "forecast_contract": manifest["forecast_contract"],
            "evaluation_note": manifest["evaluation_note"],
            "period": split,
            "cohort": cohort,
            "selected_model": LABELS[manifest["selected_on_validation"]],
            "comparison": subset_metrics.to_dict(orient="records"),
            "inspected_series": {"store_id": store, "item_id": item},
        }
        try:
            with st.spinner("Writing summary"):
                st.write(summarize_results(ROOT, payload))
        except OpenAIError as error:
            st.error(f"The summary could not be generated: {type(error).__name__}")


if __name__ == "__main__":
    main()
