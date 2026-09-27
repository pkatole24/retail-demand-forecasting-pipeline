# Repository guidance

## Review checkpoint (2026-09-26)

The repository has already been reviewed end to end. Start with `README.md`
for its historical design and findings. Inspect only the code and evidence
relevant to a new change, and repeat a full audit only when the architecture
or data contract has changed
enough to make this checkpoint obsolete.

The historical workflow builds daily store-item features from M5 sales,
calendar, and weekly prices, then evaluates lag-7, Random Forest, and a
LightGBM notebook. Its 2015-2016 results describe rolling daily scoring with
earlier holdout actuals available to later predictions. They do not establish
a fixed-origin, seven-day forecast. The baseline and model headline scores
were originally computed on different eligible row sets. Local processed
Parquet was written with Arrow 22 and failed to read with the installed Arrow
19 at this review.

Keep results from a new forecast horizon or evaluation population separate
from those historical results. Do not describe offline forecast accuracy as
measured business impact.

The current weekly extension is in `src/weekly_forecast.py` and `app.py`.
`README.md` defines its Sunday-close/next-seven-day contract; the first
completed backtest is summarized in `reports/weekly_results.md`. Its local
artifacts are regenerated under ignored `data/weekly/`. Use these current
files for questions about the weekly experiment, and the original scripts
and notebook for the daily experiment.
The 2016 weekly period was inspected before the sparse-cohort definition was
refined, so its current scores are exploratory rather than an untouched test.

## Working conventions

- Challenge weak assumptions when planning features or modeling claims.
- For proposed builds, name likely failure modes in two or three lines.
- Prefer the smallest complete modeling workflow and add capabilities in layers.
- Remove obsolete implementation paths instead of adding compatibility layers.
- Add tests for meaningful behavior, leakage boundaries, or observed bugs;
  avoid tests that merely mirror constants or assert removed code stays absent.
- Write comments for non-obvious rationale or constraints, not obvious code.
