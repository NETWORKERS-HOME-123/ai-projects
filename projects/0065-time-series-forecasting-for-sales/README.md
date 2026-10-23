# Time series sales forecasting (fixed-origin)

## What this project does
This teaching project demonstrates fixed-origin multi-step forecasting for daily sales totals. It compares a simple seasonal-naive baseline (repeat the last-week total) with a lag-feature regression model (RandomForest) that is trained to predict the total sales over a fixed horizon (one week) using only information that would have been known at the forecast origin (historical lags and planned promotions).

The code performs a rolling-origins backtest: at each origin we train a horizon-specific model using only past data, then forecast the next 7 days as a single aggregated target. Results report mean absolute error (MAE), bias (mean error), and a simple measure of seasonal failures.

## Data
The data are synthetic daily sales created inside the code: a smooth trend, weekly seasonality, a promotion effect (promotions are assumed planned and therefore known at the origin), and Gaussian noise. This small synthetic set is used so the exercise remains deterministic and self-contained.

## How to build it
1. Install the Python requirements listed in requirements.txt.
2. Run the project module (python project.py) to print a JSON summary of results.
3. Inspect the returned forecast-versus-actual snippets in the JSON for reviewer plots or further analysis (no files are written).
4. Run pytest to execute automated checks of determinism and metric floors.

## How to run
Run the project directly: python project.py
This prints JSON with the reported metrics and a small sample of forecast-versus-actual rows for inspection.

## How to test
Run pytest in the project directory: pytest -q
Tests check keys exist, metrics are in plausible ranges, a conservative performance floor is met, and that the main metric is deterministic for a fixed seed.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| model | RandomForestRegressor |
| horizon_days | 7 |
| origins_evaluated | 23 |
| mae | 14.1332 |
| baseline_mae | 20.4838 |
| neg_mae | -14.1332 |
| bias | -1.2365 |
| seasonal_failure_rate | 0.3913 |

Also returned (see `python project.py` output): `sample_forecasts` (6 items).

## Limitations
This project uses synthetic data and a single aggregated-horizon target (sum of next 7 days). It therefore does not demonstrate per-day multi-step recursive forecasting or prediction intervals. The synthetic sales process is simplified: real operational issues such as returns, reporting lags, complex event calendars (holidays) or inventory constraints are not modelled. The lag features are computed from actual historical observations — in real production you must ensure the same values (or proxies) are available at forecast time.

## Next steps
Suggested extensions: produce per-day multi-step forecasts (either direct multi-output models or recursive forecasts with careful use of predicted lags), add uncertainty estimates, include holiday calendars and special events, and validate on real sales history with proper operational feature engineering.

---

**Full case study, build steps and evaluation checks:** [Time series forecasting for sales on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0065/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
