import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def make_series(n=240, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = 0.03 * t
    season = 5.0 * np.sin(2 * np.pi * t / 30.0)
    noise = rng.normal(scale=1.0, size=n)
    return trend + season + noise


def build_windows(series: np.ndarray, window_size: int):
    X = []
    y = []
    for i in range(len(series) - window_size):
        X.append(series[i : i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def run(seed: int = 42) -> Dict[str, Any]:
    # Configuration
    series_length = 240
    window_size = 30
    # Make test horizon larger than window_size so test split can produce windows
    test_horizon = 60  # final N points held out chronologically

    # 1) Data
    series = make_series(series_length, seed=seed)

    # We will create windows but ensure windows do not cross split boundaries.
    # Split chronologically: train_end, val_end, test_end
    test_start = series_length - test_horizon
    # Ensure validation period produces at least one window: must be > window_size
    val_horizon = 31
    val_start = test_start - val_horizon
    train_series = series[:val_start]
    val_series = series[val_start:test_start]
    test_series = series[test_start:]

    X_train, y_train = build_windows(train_series, window_size)
    X_val, y_val = build_windows(val_series, window_size)
    X_test, y_test = build_windows(test_series, window_size)

    # If any split has zero rows (possible with tiny series), raise
    if X_train.size == 0 or X_val.size == 0 or X_test.size == 0:
        raise ValueError("One of the splits produced no windows; increase series length or reduce window_size/horizons.")

    # 2) Scale and model (we use an MLP on flattened windows as a lightweight sequence model proxy)
    scaler = StandardScaler()
    X_train_flat = X_train.reshape(len(X_train), -1)
    X_val_flat = X_val.reshape(len(X_val), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)

    scaler.fit(X_train_flat)
    X_train_s = scaler.transform(X_train_flat)
    X_val_s = scaler.transform(X_val_flat)
    X_test_s = scaler.transform(X_test_flat)

    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 16),
        activation='relu',
        solver='adam',
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        max_iter=500,
        random_state=seed)

    mlp.fit(np.vstack([X_train_s, X_val_s]), np.concatenate([y_train, y_val]))

    # 3) Baselines: persistence (predict last value in window) and Ridge on flattened lags
    y_pred_mlp = mlp.predict(X_test_s)
    y_pred_persist = np.array([w[-1] for w in X_test])

    ridge = Ridge(alpha=1.0, random_state=seed)
    ridge.fit(X_train_flat, y_train)
    y_pred_ridge = ridge.predict(X_test_flat)

    # Metrics
    mae_mlp = float(mean_absolute_error(y_test, y_pred_mlp))
    r2_mlp = float(r2_score(y_test, y_pred_mlp))

    mae_persist = float(mean_absolute_error(y_test, y_pred_persist))
    r2_persist = float(r2_score(y_test, y_pred_persist))

    mae_ridge = float(mean_absolute_error(y_test, y_pred_ridge))
    r2_ridge = float(r2_score(y_test, y_pred_ridge))

    # Example window and prediction (first test row) to show inputs and next-step
    example_window = X_test[0].tolist()
    example_pred = float(y_pred_mlp[0])

    out = {
        "seed": int(seed),
        "window_size": int(window_size),
        "horizon_test_points": int(len(y_test)),
        "test_r2_mlp": r2_mlp,
        "test_mae_mlp": mae_mlp,
        "test_r2_persistence": r2_persist,
        "test_mae_persistence": mae_persist,
        "test_r2_ridge": r2_ridge,
        "test_mae_ridge": mae_ridge,
        "example_window": example_window,
        "example_prediction_mlp": example_pred,
        "notes": "Model is an MLP on flattened windows used as a lightweight proxy for sequence models due to environment constraints."
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
