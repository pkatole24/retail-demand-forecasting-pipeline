# M5 demand forecasting workbench

This repository contains two distinct M5 retail-demand experiments. The original
one scores **daily** store-item demand with a lag-7 baseline, Random Forest,
and a LightGBM notebook. The newer one issues a forecast each Sunday for the
**total units sold during the following Monday-Sunday week**. Results from
these tasks must not be compared as if they were the same forecast.

## Weekly forecasting question

At the end of a week, which store-item demand forecasts are accurate enough
to inform a replenishment discussion, and where do models systematically
over- or underforecast? This is an offline forecast-quality demonstration.
It does not simulate orders or measure inventory savings.

The weekly experiment uses a deterministic 200-series portfolio chosen with
information available by the end of 2014: 100 highest-volume active series
and 100 active series with sales on 1-10% of 2014 days. Each chosen series
has at least ten units sold in 2014 and a first recorded sale by the end of
2013. Selection does not look at validation or test demand. The selected
portfolio is saved as `data/weekly/selected_series.csv`.

Every row represents a Sunday forecast origin and one store-item series.
The target is the sum of the next seven daily sales values. Demand features
use sales through that Sunday only. Price features use prices observed no
later than that Sunday; a missing-price flag is retained instead of dropping
the row. Calendar week and month are known at forecast issuance.

| Comparison | Forecast for the next week |
| --- | --- |
| Last week | Sales during the seven days ending at origin |
| Four-week average | Mean of the four prior seven-day totals |
| CrostonOptimized | Sum of seven daily intermittent-demand forecasts fitted through origin |
| LightGBM | Global regression model predicting the seven-day total directly |

Weeks entirely before 2015 train LightGBM; weeks entirely in 2015 form the
validation set; and weeks entirely in 2016 through April 24 form the test
period. Cross-year weeks are excluded. Model choice uses validation WAPE. The
LightGBM model is then fitted on training plus validation history before its
test forecasts are made. All four comparisons use identical validation and
test rows. MAE, aggregate WAPE, and signed bias are reported for the full
portfolio and the two demand groups.

During development, an initial 2016 readout was inspected before the sparse
portfolio definition was tightened from 5-40% to 1-10% selling days. The
2016 scores are therefore exploratory, not an untouched final confirmation.
An independent later period would be needed for that claim.

## Run locally

1. Obtain the M5 Forecasting Accuracy files and place
   `sales_train_validation.csv`, `calendar.csv`, and `sell_prices.csv` in
   `data/raw/`. These files are ignored by Git.
2. Use Python 3.13, create an environment, and install the tested versions
   in `requirements.txt`. On Windows, for example:

   ```powershell
   py -3.13 -m venv .venv
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. From the repository root, build the weekly experiment and start the app:

   ```powershell
   .venv\Scripts\python.exe -m src.weekly_forecast
   .venv\Scripts\python.exe -m streamlit run app.py
   ```

   The pipeline writes `predictions.parquet`, `metrics.csv`,
   `selected_series.csv`, `lightgbm.joblib`, and `run.json` under
   `data/weekly/`. The manifest records source-file SHA-256 hashes, package
   versions, row counts, the forecast contract, and the validation-selected
   model. The app reads these saved artifacts rather than retraining when a
   filter changes. Run `python -m pytest -q` for the focused boundary tests.

The optional **Explain these results** button uses the OpenAI API to
summarize computed metrics. To enable it, set `OPENAI_API_KEY` and
`OPENAI_MODEL` in the ignored `.env` file (see `.env.example`). The LLM
receives selected aggregate metrics and series identifiers, and cannot alter
model predictions or scores. The workbench runs without an API key.

## Recorded local backtest

The first completed run's results are recorded in
`reports/weekly_results.md`; rerun the command above to regenerate the full
artifacts. The result is nuanced: LightGBM wins on aggregate validation WAPE,
while the four-week average is stronger on the sparse-demand group. Croston
does not win despite being designed for intermittent demand. This is useful
evidence about this data and horizon, not a reason to hide the model.

## Historical daily experiment

The original workflow remains in `src/build_tables.py`,
`src/build_analytic_table.py`, `src/features.py`, `src/train_baseline.py`,
`src/train_model.py`, and `notebooks/lightgbm.ipynb`. It reshapes M5's wide
daily sales to a 58.3-million-row fact table, aligns weekly prices through
the calendar, constructs lag and rolling features, and scores a lag-7
baseline, Random Forest, and LightGBM. The original README recorded baseline
MAE/WAPE of **1.14/0.90** and Random Forest **0.94/0.74**.

Those historical scores describe daily rolling scoring: later holdout dates
can use earlier holdout actuals. They do not demonstrate a single-origin
480-day forecast. The baseline was first scored on all eligible holdout rows
and RF on a sampled, feature-complete subset. Those original headline scores
should not be presented as a same-sample model improvement.
