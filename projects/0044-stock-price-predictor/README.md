# Stock price predictor

## What this project does

This educational project demonstrates a simple approach to short-horizon stock-price forecasting using synthetic adjusted-price time series. It emphasises chronological evaluation (walk-forward), compares a lag-feature regression model to two naive baselines (last-price and simple-return extrapolation), and reports mean absolute error (MAE) and R2 at multiple horizons.

This is not investment advice, trading execution code, or a backtest using licensed market data. Instead the dataset is a small synthetic stand-in that mimics key statistical features of equity returns for educational purposes only.

## Data

The data are synthetic adjusted prices generated inside the code with a geometric Brownian motion (GBM) process. From that price series we build lag features (last price, prior returns), short moving averages and target prices at horizons 1, 5 and 10 days. For reproducibility the generator uses a fixed random seed passed to the run() function.

## How to build it

1. Create the synthetic adjusted-price time series (GBM) and build a Pandas DataFrame of features and targets.
2. Prepare lag features and rolling statistics and align them so every row only uses information available at the timestamp of prediction.
3. Implement a walk-forward (expanding-window) evaluation: fit a Ridge regression on past data and predict the next evaluation block; compute MAE and R2 per horizon and compare with baselines.
4. Package results into a small JSON-like dictionary and print them from an if __name__ == "__main__" block.

## How to run

Run the project file with Python 3.11+. It prints a JSON object with the metrics. Example:

python project.py

You can also import run from project and call run(seed=42) programmatically.

## How to test

This repository includes pytest tests. From the project root run:

pytest -q

The tests check presence of expected keys, conservative metric floors, valid ranges and determinism of the main metric when called with the same seed.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| mae_horizon_1 | 1.0004 |
| r2_horizon_1 | 0.9821 |
| baseline_last_r2_h1 | 0.9875 |
| baseline_last_mae_h1 | 1.2272 |
| baseline_return_r2_h1 | 0.9886 |
| baseline_return_mae_h1 | 1.1786 |
| mae_horizon_5 | 2.3329 |
| r2_horizon_5 | 0.8940 |
| baseline_last_r2_h5 | 0.9636 |
| baseline_last_mae_h5 | 2.0488 |
| baseline_return_r2_h5 | 0.9411 |
| baseline_return_mae_h5 | 2.6039 |
| mae_horizon_10 | 3.2970 |
| r2_horizon_10 | 0.8097 |
| baseline_last_r2_h10 | 0.9389 |
| baseline_last_mae_h10 | 2.7587 |
| baseline_return_r2_h10 | 0.8342 |
| baseline_return_mae_h10 | 4.5031 |
| avg_transaction_cost_per_prediction | 0.0500 |
| note | Synthetic GBM used. Model: Ridge regression. Chronological walk-forward evaluation. Educational only. |

## Limitations

- The dataset is synthetic and small. Real market data include corporate actions, market microstructure effects and regime changes that are not modelled here.
- No transaction or slippage modelling beyond a simple illustrative cost parameter; a proper backtest requires order execution modelling and realistic costs.
- The ML model used here is a simple Ridge regression; more advanced models and careful feature engineering may be required in real applications.
- Chronological validation here is simplified; production systems should use more thorough walk-forward hyperparameter tuning and out-of-sample monitoring.

## Next steps

- Replace the synthetic series with licensed historical adjusted prices and ensure timestamps for feature availability are recorded.
- Extend features (fundamentals, intraday liquidity measures) and add robust cross-validation for time-series (nested walk-forward).
- Implement a realistic trading-cost model and a portfolio-level evaluation (Sharpe, drawdown) if the goal is strategy assessment.
- Experiment with more expressive models (gradient boosting, recurrent models) while guarding against overfitting and data leakage.

---

**Full case study, build steps and evaluation checks:** [Stock price predictor on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0044/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
