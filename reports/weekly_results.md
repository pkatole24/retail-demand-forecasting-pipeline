# Seven-day store-item backtest (local run, 2026-09-26)

Forecasts were issued at Sunday close for the following Monday-Sunday total.
The portfolio was selected using 2014 sales only: 100 high-volume and 100
intermittent series (sales on 1-10% of 2014 days). Validation contains 51
weeks in 2015 and 10,200 store-item forecasts; the 2016 test period contains
16 weeks in 2016 and 3,200 forecasts. Every model below was scored on the
same rows within each period. WAPE is total absolute unit error divided by
total actual units for the displayed group.

| Model | Validation WAPE, all | Test WAPE, all | Test WAPE, high volume | Test WAPE, sparse |
| --- | ---: | ---: | ---: | ---: |
| Last week | 0.235 | 0.214 | 0.200 | 0.485 |
| Four-week average | 0.264 | 0.231 | 0.221 | **0.428** |
| CrostonOptimized | 0.306 | 0.247 | 0.235 | 0.493 |
| LightGBM | **0.229** | **0.210** | **0.195** | 0.510 |

LightGBM was selected using aggregate validation WAPE, then refitted using
training plus validation weeks before test forecasting. On test it reduced
aggregate WAPE by about **1.6% relative to last week**, much smaller than the
historical daily-model gain and not directly comparable with it. The
four-week average performed best on the sparse group. CrostonOptimized did
not win even there, which is a useful negative result: a method designed
for intermittent observations is not automatically best for a seven-day
total. LightGBM's sparse-group weakness is a reason to examine cohort-aware
model selection before suggesting one model for all items.

An initial 2016 readout was inspected before the sparse-cohort range was
tightened from 5-40% to 1-10% selling days. The displayed 2016 results are
therefore exploratory and need confirmation on a later, unseen period.

The portfolio is fixed from 2014 behavior, so it does not represent every
M5 item, new products, or a live catalog. The test covers only 16 weekly
origins. These are forecast-quality findings; no order quantities, stockouts,
or realized inventory costs were measured. Full predictions, cohort metrics,
input hashes, and environment details are regenerated locally under
`data/weekly/` by `python -m src.weekly_forecast`.
