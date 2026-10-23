# Demand forecasting system

## What this project does

This project demonstrates a small, reproducible demand-forecasting pipeline for many SKUs (items) with intermittent sales, promotions and availability (stockout) flags. It compares a lag-feature regression model (RandomForestRegressor) against two simple baselines: a seasonal-naive baseline that repeats the same weekday's demand from the past week, and an intermittent-demand baseline that uses the historical mean of positive demand. The pipeline emphasises a single-origin multi-step forecast (a horizon made at one time) and uses rolling-origin evaluation to compute mean absolute error (MAE) and bias per item, with additional breakdown for stockout periods.

This is intended as a teaching project for inventory planning scenarios: how to prepare dated item-level sales with availability flags, how to build lag features for multi-step forecasts, how to compare sensible baselines, and how to report planner-friendly forecast outputs with caveats.

## Data

The data are synthetic and generated in code. The generator simulates multiple items across daily dates with: a baseline demand level per item, a weekly seasonality, occasional promotions that temporarily amplify demand, and intermittent stockouts that force observed sales to zero despite latent demand. The generator also produces an availability flag and a replenishment dynamic so some stockouts persist for several days. The synthetic data is deliberately simple so the exercises focus on modelling choices and evaluation methodology.

## How to build it

1. Prepare the data — Prepare dated item-level sales with availability flags and a fixed forecast horizon.
2. Build the experiment — Compare lag-feature regression with seasonal-naive and intermittent-demand baselines where appropriate.
3. Evaluate the result — Use rolling-origin MAE and bias by item, examining stockout periods separately.
4. Define a reviewable output — Provide planner-reviewed forecasts with horizon labels and stockout caveats.

## How to run

Run the project module. It exposes run(seed: int = 42) -> dict which generates the synthetic dataset, trains the model, runs forecasts from a single origin for a fixed horizon, computes baselines and evaluation metrics, and returns a JSON-serialisable result. The module also prints the result when executed as a script.

## How to test

Tests are provided using pytest. They check that run returns expected keys, that numeric metrics are in reasonable ranges, that a conservative MAE floor is met on the held-out horizon, and that repeated calls with the same seed are deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| mae_holdout | 3.1771 |
| bias_holdout | -0.8890 |
| seasonal_naive_mae | 7.1042 |
| intermittent_mae | 5.6219 |
| stockout_mae | 18.7939 |
| horizon | 30 |
| n_items | 8 |

Also returned (see `python project.py` output): `forecasts_sample` (3 items).

## Limitations

This is a compact educational example. The synthetic data is simplified: it does not model complex replenishment policies, lead times, or realistic cross-item cannibalisation. The regression model performs recursive forecasting (using its own prior step predictions as inputs for later steps), which is only one practical strategy among several (direct multi-output models or probabilistic forecasting are alternatives). Prediction intervals are not provided.

## Next steps

Possible extensions include adding probabilistic forecasts (prediction intervals), modelling lead times and replenishment policies explicitly, using direct multi-horizon models, and experimenting with feature sets such as promotions encoding, holidays and longer seasonalities.

---

**Full case study, build steps and evaluation checks:** [Demand forecasting system on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0059/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
