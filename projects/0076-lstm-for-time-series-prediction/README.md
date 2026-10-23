# LSTM for time series prediction

## What this project does

This small teaching project demonstrates how to build a chronological time-series forecasting pipeline with windowed inputs, careful train/validation/test splits and a sequence model. The original starter used a RandomForest on lag columns; here we show a simple sequence-to-one model trained on fixed-length windows, compare it with a persistence baseline and a lag-feature linear regressor, and highlight limitations. Note: the environment for this assignment restricts deep-learning libraries (PyTorch/TensorFlow). To keep the exercise runnable with only the approved libraries, the code trains a small feed-forward network (MLP) on flattened windows and uses sklearn's early stopping. The README is honest about that substitution and explains why windowing and split strategy still carry the same lessons as training an LSTM.

## Data

The data are synthetic and generated in code: 240 daily values composed from a small linear trend, a monthly-ish sinusoidal seasonality and Gaussian noise. This is a compact stand-in for real sensor/time-series data and is built with numpy inside the project (no downloads or file I/O).

## How to build it

1. Create a reproducible synthetic series (trend + seasonality + noise) of length 240.
2. Build non-overlapping windows and next-step targets so windows do not cross chronological split boundaries (train/val/test).
3. Scale features using StandardScaler fitted on training windows; train a small MLPRegressor with early_stopping=True on train/val.
4. Evaluate on the held-out final windowed test set; compare with persistence and a Ridge regression on lag features.

## How to run

Run the project module; it exposes run(seed: int = 42) and prints a JSON report when run as a script:

python project.py

run(seed) returns a JSON-serialisable dict containing the main held-out metrics, an example input window and its next-step prediction, and the tested horizon.

## How to test

Use pytest. The provided test file imports run from project.py and checks keys exist, metrics fall inside valid ranges, determinism for a fixed seed, and a conservative metric floor on the main held-out metric.

## Results

<!-- RESULTS -->
Measured by running `python project.py` (seed 42) on 2026-10-23; the tests in `test_project.py` assert the floor.

| Metric | Value |
|---|---|
| seed | 42 |
| window_size | 30 |
| horizon_test_points | 30 |
| test_r2_mlp | 0.8677 |
| test_mae_mlp | 1.0226 |
| test_r2_persistence | 0.7358 |
| test_mae_persistence | 1.4200 |
| test_r2_ridge | 0.9042 |
| test_mae_ridge | 0.8480 |
| example_prediction_mlp | 6.2405 |
| notes | Model is an MLP on flattened windows used as a lightweight proxy for sequence models due to environment constraints. |

Also returned (see `python project.py` output): `example_window` (30 items).

## Limitations

- The code does not use an actual LSTM implementation because PyTorch/TensorFlow are not available in the constrained environment. Instead we flatten windows and train an MLP with early stopping. This reproduces the important engineering lessons (window creation, chronological splits, scaling, baselines) but is not an LSTM.
- Forecasts are one-step-ahead from a window. Multi-step forecasting or recursive forecasting strategies are not explored here.
- No prediction intervals are provided.

## Next steps

- Replace the MLP with a proper LSTM/GRU implementation (PyTorch or TensorFlow) when such libraries are available.
- Implement multi-step forecasting from a single origin and evaluate recursive vs direct strategies.
- Add uncertainty estimates (quantile regression or ensembles) and richer evaluation over different forecast horizons.

---

**Full case study, build steps and evaluation checks:** [LSTM for time series prediction on networkershome.com](https://www.networkershome.com/ai-projects-courses/projects/0076/)

Built by [Networkers Home](https://www.networkershome.com) · Learn the skills in the [AI Engineer Course](https://www.networkershome.com/ai-engineer-course-india/) · Part of [100 AI projects with code](../../)
