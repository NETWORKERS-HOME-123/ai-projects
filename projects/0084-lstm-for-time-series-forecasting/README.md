# Multi-step forecasting: direct vs recursive (linear models)

## What this project does

This small teaching project demonstrates multi-step time-series forecasting and compares two operational strategies: direct forecasting (one model per horizon) and recursive forecasting (one-step model iterated forward). It constructs a synthetic seasonal time series, prepares chronological windows, trains simple linear models, and evaluates forecasts from fixed issue dates (backtest origins). Results report horizon-specific mean absolute error (MAE) for both strategies and a seasonal-naive baseline. The code is deterministic and simple so you can study the mechanics of multi-step evaluation.

## Data

The data are synthetic: a single univariate daily series with a linear trend, a yearly-like seasonal component, and Gaussian noise. No external data are used. The series is generated inside the code and split chronologically into training and test segments.

## How to build it

1. Create a new Python 3.11+ virtual environment and activate it.
2. Save the provided files (project.py, test_project.py, README.md) into a project folder.
3. Install dependencies: pip install -r requirements.txt
4. Run the project with Python to produce the JSON summary and review-only report.

## How to run

Run the main script directly:

python project.py

It prints a JSON object containing horizon-specific MAE numbers, a short review table encoded as strings, and the list of issue dates used in backtesting.

## How to test

This repository includes pytest tests. Run them with:

pytest -q

Tests check that run(seed) returns expected keys, that MAE metrics are non-negative, that a conservative floor is met for the primary metric, and that calling run(42) twice is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| mae_direct_mean | 0.2022 |
| mae_recursive_mean | 0.1987 |
| mae_naive_mean | 0.5019 |
| mae_direct_h1 | 0.1808 |
| mae_recursive_h1 | 0.1748 |
| mae_naive_h1 | 0.2783 |
| mae_direct_h2 | 0.1470 |
| mae_recursive_h2 | 0.1412 |
| mae_naive_h2 | 0.3167 |
| mae_direct_h3 | 0.2556 |
| mae_recursive_h3 | 0.2353 |
| mae_naive_h3 | 0.5060 |
| mae_direct_h4 | 0.1676 |
| mae_recursive_h4 | 0.1906 |
| mae_naive_h4 | 0.4422 |
| mae_direct_h5 | 0.2379 |
| mae_recursive_h5 | 0.2316 |
| mae_naive_h5 | 0.5322 |
| mae_direct_h6 | 0.2113 |
| mae_recursive_h6 | 0.2177 |
| mae_naive_h6 | 0.6623 |
| mae_direct_h7 | 0.2153 |
| mae_recursive_h7 | 0.2000 |
| mae_naive_h7 | 0.7754 |
| review_report_csv | issue_idx,h1,h2,h3,h4,h5,h6,h7
300,0.283,0.148,0.194,0.225,0.074,0.079,0.374
307,0.222,0.169,0.015,0.148,0.323,0.202,0.289
314,0.373,0.048,0.164,0.129,0.120,0.064,0.115
321,0.105,0.314,0.677,0.302,0.388,0.384,0.249
328,0.166,0.050,0.030,0.192,0.092,0.144,0.184
335,0.111,0.227,0.252,0.111,0.297,0.343,0.007
342,0.176,0.124,0.267,0.014,0.575,0.470,0.173
349,0.009,0.095,0.445,0.219,0.036,0.005,0.330 |
| note_model | ridge_linear_models_direct_and_one_step_recursive |

Also returned (see `python project.py` output): `issue_origin_indices` (8 items).

## Limitations

This project uses very simple linear models (Ridge regression) for clarity and pedagogical purposes rather than an LSTM. A linear modelling approach is fast and easy to inspect but will not capture complex non-linear sequential dynamics as an LSTM might. Also, models are trained on fixed windows of historical observations (an autoregressive-style feature vector), and there is no uncertainty quantification or prediction intervals provided.

## Next steps

- Replace the linear models with a small RNN/LSTM implementation (for example using PyTorch) to study learned sequential representations.
- Add probabilistic forecasting or simple bootstrap-based intervals.
- Experiment with different backtest origin schedules and longer horizons to inspect stability of direct vs recursive strategies.

---

**Full case study, build steps and evaluation checks:** [LSTM for time series forecasting on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0084/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
