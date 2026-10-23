# Anomaly detection in time series

## What this project does

This small project builds a simple operational anomaly-detection pipeline for a single sensor time series. It demonstrates how to (a) prepare timestamped data with annotated events and missing-data intervals, (b) fit a rolling baseline and set residual thresholds using training data only, (c) evaluate detection performance on held-out periods using event-level metrics, and (d) produce a concise review timeline showing alerts and technician feedback for operators.

The code uses a lightweight forecasting baseline (a linear trend plus daily seasonality) and performs iterative one-origin forecasting across the test period so the model does not peek at future observations when forming lagged inputs.

## Data

The dataset is synthetic and generated inside the code. It simulates 180 days of hourly sensor readings with trend, daily seasonality, occasional missing-data blocks, and annotated anomaly events (spikes and sustained shifts). The annotations are generated together with the signal so the evaluation compares alerts to ground-truth event windows.

## How to build it

1. Create a Python 3.11+ virtual environment and activate it.
2. Install dependencies: pip install -r requirements.txt
3. Inspect project.py to understand the pipeline: data generation, baseline fitting, thresholding, forecasting and evaluation.
4. Run the module to print evaluation metrics and a small review timeline.

## How to run

Run the pipeline directly: python project.py

It will print a JSON object with the main metrics and a small review timeline snippet.

## How to test

Run pytest in the project directory: pytest -q

The tests check that run(seed=42) returns the expected keys, metrics are sensible, the main metric comfortably exceeds a conservative floor, and the pipeline is deterministic for the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| event_recall | 1 |
| false_alerts_per_day | 4.8700 |
| mean_detection_delay_hours | 0 |
| alerts_count | 299 |
| annotated_events_considered | 3 |
| threshold_used | 1.1464 |

Also returned (see `python project.py` output): `review_timeline_preview` (8 items).

> **Read the `event_recall` value with care:** the data here is small and synthetic with a planted, learnable structure, so a near-perfect score shows the pipeline works, not that the method would reach this on real data. See **Limitations**.

## Limitations

The data is synthetic and much simpler than real industrial telemetry: there is a single sensor, noise characteristics are stationary outside events, and annotations are generated programmatically. The baseline model is a simple linear trend plus daily seasonality; it is not a complex probabilistic model and so prediction intervals are heuristic (percentile of training residuals). Multivariate signals, richer seasonality, and contextual metadata are out of scope for this exercise.

## Next steps

- Replace the simple baseline with an online model (e.g. Kalman filter or state-space model) and estimate principled predictive intervals.
- Add multivariate context (other sensors) and train a conditional baseline.
- Implement alert scoring that downweights brief isolated spikes if they coincide with known operational events.
- Build a small UI for the review timeline allowing technicians to annotate alerts during review.

---

**Full case study, build steps and evaluation checks:** [Anomaly detection in time series on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0054/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
