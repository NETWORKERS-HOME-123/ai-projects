# Customer lifetime value predictor

## What this project does

This small project demonstrates how to build a reproducible pipeline that predicts a one‑year customer lifetime value (CLV) from information available at a fixed observation cutoff. It emphasises comparing a simple signup‑month group average baseline with a supervised regression model, and evaluating absolute errors by signup group and value band. The code is synthetic but structured to illustrate common practical considerations: temporal drift, censored histories, value skew and planning‑level aggregation with uncertainty.

## Data

The dataset is synthetic and generated inside the code using scikit‑learn's make_regression plus engineered signup month and censoring effects. Each row represents an anonymised account with numeric features observable at the cutoff, a signup month group (0–11), and a future 12‑month revenue label. The synthetic construction intentionally includes noise, a loglike skew in target values and occasional censored (reduced) observations to mimic returns/cancellations.

## How to build it

1. Create the synthetic dataset (features, signup month group, target revenue) and apply a deterministic random seed.  
2. Split into train and held‑out test sets with a fixed random_state.  
3. Fit a regression baseline that uses signup month group mean revenue and a RandomForestRegressor that uses the available features.  
4. Evaluate absolute error (MAE) and R2 overall, plus MAE by signup month group and by value band. Return aggregate planning metrics and a simple uncertainty estimate.

## How to run

Run the project file with Python 3.11+. The script exposes a run(seed: int = 42) function; executing the file prints a JSON summary. No external data or network access is required.

## How to test

Install the required packages (see requirements). Use pytest to run the included tests which check keys, metric ranges, determinism and a conservative performance floor on the held‑out R2.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| r2 | 0.6474 |
| mae_model | 4537.2775 |
| mae_baseline_signup_group | 8226.1828 |
| group_mae_mean | 4547.9097 |
| band_mae_low | 6304.6143 |
| band_mae_mid | 2507.0784 |
| band_mae_high | 4865.7068 |
| planning_total_pred | 25074162.3108 |
| planning_total_uncertainty_se | 6985.4216 |
| n_test | 750 |
| model | RandomForestRegressor |

## Limitations

This is a synthetic demonstration. Real CLV estimation requires detailed transaction histories, explicit treatment of returns/cancellations, business margins rather than gross revenue, and careful treatment of censoring and survival. The model here is intentionally simple and not suitable for direct roll‑out to production without real data and domain validation.

## Next steps

Possible extensions include: calibrating uncertainty with quantile regression or bootstrap aggregation, using survival models for censored lifetimes, adding temporal cross‑validation, incorporating business margin features and return indicators, and validating on real transaction logs with domain experts.

---

**Full case study, build steps and evaluation checks:** [Customer lifetime value predictor on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0048/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
