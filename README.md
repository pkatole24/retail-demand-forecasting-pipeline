# M5 Operations Forecasting Pipeline

## Project Goal
Build an end-to-end pipeline that transforms raw retail sales data into forecasting-ready features and evaluates demand forecasting models using time-aware validation.

---

## Data Source
M5 Forecasting – Accuracy dataset (Kaggle), containing:
- daily item-level sales by store
- calendar metadata
- weekly item prices by store

---

## Pipeline Overview

### 1. Raw Data → Structured Fact Tables

**fact_sales.parquet**
- Grain: `(date, store_id, item_id)`
- Columns:
  - `date`
  - `store_id`
  - `item_id`
  - `units_sold`

Daily sales are reshaped from wide format (`d_1 … d_1913`) to long format and joined with the calendar table to attach real dates.

Quality checks enforced:
- no duplicate keys
- no negative sales
- no missing identifiers

---

### 2. Align Weekly Prices to Daily Sales

Weekly prices cannot be joined directly to daily sales.  
The calendar table is used as a bridge via `wm_yr_wk`.

**daily_sales_prices.parquet**
- Grain: `(date, store_id, item_id)`
- Added columns:
  - `wm_yr_wk`
  - `sell_price`

Prices are joined using `(store_id, item_id, wm_yr_wk)` with a left join to preserve the full sales history.

#### Missing Prices
Some rows contain missing `sell_price` values. This is expected due to:
- items not priced in certain store–weeks
- products introduced after the start of the series
- incomplete early-period price coverage

Missing prices are handled explicitly during feature engineering rather than dropped.

---

### 3. Feature Engineering

Features are computed at daily grain using only historical information to avoid leakage.

**Demand history**
- `lag_7`, `lag_28`
- `roll_mean_7`, `roll_mean_28`

**Calendar**
- `day_of_week`

**Price dynamics**
- `price_lag_1w`
- `price_change_1w`
- `price_roll_mean_4w`

All lag and rolling features are computed within `(store_id, item_id)` groups after sorting and verifying monotonic time order.

**daily_features.parquet**
- Final modeling-ready dataset

---

## Data Storage Decisions

### Why Parquet
Processed tables are stored in Parquet format because it:
- preserves data types (dates, integers)
- loads significantly faster than CSV for large datasets
- is the industry standard for analytics and ML pipelines

This is critical given the dataset contains tens of millions of daily observations.

---

## Modeling Approach

### Baseline Model
A simple lag-based baseline is used to establish a defensible benchmark.

- Prediction: `y_hat = lag_7`
- Validation: strict time-based split
- Metrics:
  - MAE (interpretability)
  - WAPE (operations relevance)

**Baseline performance**
- MAE: **1.14**
- WAPE: **0.90**

---

### Machine Learning Model
A tree-based model is used to capture nonlinear relationships in tabular data.

**Model**
- RandomForestRegressor

**Features**
- demand lags and rolling means
- calendar effects
- price dynamics

**Validation**
- same time-based split as baseline
- train on historical period, test on future period

**Model performance**
- MAE: **0.94**
- WAPE: **0.74**
- ~**17–18% error reduction** over baseline

Feature importance shows:
- short-term rolling demand as the dominant signal
- medium-term trends and weekly seasonality
- prices providing incremental explanatory power

---

## Design Choices and Rationale
- **Tree-based models** were chosen over neural networks due to superior performance on tabular, lag-based data and lower tuning complexity.
- **No normalization** was applied, as tree models are scale-invariant.
- **High-cardinality identifiers** (e.g., item_id) were not one-hot encoded; demand history features capture item-level behavior more robustly.
- **Time-aware validation** was enforced throughout to prevent leakage.

---

## How to Run
1. Create and activate the conda environment.
2. Install dependencies from `requirements.txt`.
3. Run scripts in order:
   - `build_tables.py`
   - `build_analytic_table.py`
   - `features.py`
   - `train_baseline.py`
   - `train_model.py`

---
