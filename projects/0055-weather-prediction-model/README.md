# Weather baseline: 24-hour temperature regression

## What this project does

This project builds a small, reproducible baseline model that predicts daily maximum temperature 24 hours ahead for a single station. It is an educational pipeline that demonstrates how to (1) prepare station-like tabular observations, (2) build a simple regression baseline, (3) evaluate against simple meteorological baselines (persistence and seasonal climatology) using a chronological split, and (4) publish a short, reviewable forecast comparison with clear limitations and links to official warnings.

This is not an operational forecast system and it must not be used for safety decisions. It is a teaching example for final-year students to practise model-building and evaluation that accounts for chronology and extreme-period assessment.

## Data

The code constructs a small synthetic station-like dataset (daily observations for ~3 years) as a faithful stand-in for a single weather station. Each row contains a date and simple predictors that would plausibly be available at forecast issue time (previous-day temperature, previous-day humidity, wind speed, and day-of-year). The target is the next-day maximum temperature (24-hour horizon). The dataset is synthetic (generated deterministically from a seed) so the example is fully self-contained and deterministic.

## How to build it

1. Install the required packages listed in requirements.txt. 2. Open project.py and inspect the run(seed=42) function to see data construction, baselines and the regression pipeline (StandardScaler + LinearRegression). 3. Run the script to produce JSON metrics (it prints them). 4. Read tests in test_project.py to understand required metrics and determinism checks.

## How to run

Run the module directly with Python 3.11+: python project.py. The script calls run() and prints JSON with evaluation metrics for the held-out chronological test set.

## How to test

Run pytest in the project directory: pytest -q. Tests check that run(42) returns expected keys, that values are in sensible ranges, that a conservative R2 floor is passed on the held-out set, and that the main metric is deterministic.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| horizon_hours | 24 |
| model | LinearRegression |
| n_samples | 1200 |
| test_size | 240 |
| mae | 2.2440 |
| r2 | 0.8695 |
| persistence_mae | 2.3284 |
| climatology_mae | 2.2937 |
| extreme_low_mae | 3.2226 |
| extreme_high_mae | 4.0620 |
| notes | Synthetic station-like data; baselines: persistence (yesterday) and seasonal climatology. See official warnings from national services (e.g. https://mausam.imd.gov.in/) for real decisions. |

## Limitations

- The dataset is synthetic and from one virtual station; real station metadata issues (sensor changes, missing observations, quality control) are not addressed.  
- The model is a simple linear regression baseline; it is intentionally lightweight for pedagogic clarity.  
- This does not replace official forecasts or warning services. For real decisions, consult an official meteorological agency and their severe-weather warnings (for India, see the India Meteorological Department: https://mausam.imd.gov.in/).

## Next steps

- Replace synthetic data with real station observations and ensure each predictor is available at forecast issue time.  
- Try non-linear models, lagged features, and probabilistic forecasts.  
- Validate with longer multi-year backtests, cross-validation that respects seasons, and test against official forecasts where available.

---

**Full case study, build steps and evaluation checks:** [Weather prediction model on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0055/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
