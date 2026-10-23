# Time series forecasting with an ARIMA( p,d,0 )-style model

## What this project does
This small teaching project demonstrates how to fit a simple ARIMA-style model for univariate time-series forecasting using only numpy, pandas and scikit-learn. It focuses on documented differencing (d), fitting an autoregressive component (p) on the differenced series, generating multi-step forecasts from a fixed origin, and comparing those forecasts (and prediction intervals) to a naive last-value baseline. It also computes horizon-specific mean absolute errors and a simple interval coverage diagnostic.

## Data
The data are synthetic: a regularly spaced daily series combining a linear trend, a seasonal weekly component and Gaussian noise. The series is generated inside the code using numpy; no external data downloads are used. The final 30 days are held out as the evaluation period.

## How to build it
1. Create a reproducible synthetic daily series (trend + weekly seasonality + noise) and reserve the final 30 points for testing.
2. Difference the training series with documented order d (we use d=1), and build lagged features for an AR(p) model (we use p=5).
3. Fit a linear regression on the differenced (AR) features. Use the in-sample one-step residuals to estimate forecast uncertainty and generate multi-step forecasts from the fixed origin by iterating the AR recursion and integrating the differenced forecasts back to the original scale.
4. Compute evaluation metrics: MAE by horizon (1, 7, 14, 30), overall MAE, R2 on the 30-day holdout, interval coverage at nominal 95%, and compare to a naive last-value baseline.

## How to run
1. Ensure Python 3.11+ is installed.
2. Install dependencies: pip install -r requirements.txt
3. Run the example directly: python project.py
4. The script prints a JSON object of metrics to stdout.

## How to test
1. Install test dependencies (same requirements include pytest).
2. Run pytest in the project directory: pytest -q
3. Tests check returned keys, reasonable metric ranges, a conservative metric floor and determinism for the main metric.

## Results
<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| model | ARIMA-like AR(p,d,0) via linear regression |
| d | 1 |
| p | 5 |
| mae | 0.8400 |
| naive_mae | 1.0525 |
| r2 | 0 |
| coverage_95 | 1 |
| mae_h1 | 0.0573 |
| mae_h7 | 0.6723 |
| mae_h14 | 0.8634 |
| mae_h30 | 0.8400 |
| resid_std | 0.5820 |

## Limitations
This project deliberately implements an ARIMA( p,d,0 )-style model: differencing (d) plus an autoregressive component (p) fitted by ordinary least squares. It does not fit a full ARIMA(p,d,q) with an explicit MA(q) term using packages like statsmodels. Prediction intervals are approximate: they assume the one-step residuals are an adequate estimator of uncertainty and scale with horizon using a simple sqrt(h) factor. The synthetic data are small and idealised; real-world time series often require more careful diagnostics and specialised tools.

## Next steps
- Replace the linear-regression AR approximation with a full ARIMA or SARIMA implementation (for example using statsmodels) for proper MA terms and likelihood-based inference.
- Improve interval estimation with analytic propagation or bootstrap of residuals.
- Add automated order selection (AIC/BIC) and residual diagnostics (Ljung–Box) for more rigorous model choice.

---

**Full case study, build steps and evaluation checks:** [Time series forecasting with ARIMA on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0025/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
