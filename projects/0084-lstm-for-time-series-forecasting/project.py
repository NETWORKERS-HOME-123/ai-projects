import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error


def _make_series(n=360, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(n)
    trend = 0.01 * t
    seasonal = 0.8 * np.sin(2 * np.pi * t / 30.0)  # monthly-ish
    noise = rng.normal(scale=0.3, size=n)
    return trend + seasonal + noise


def _make_windows(series: np.ndarray, lookback: int, max_horizon: int):
    X = []
    Ys = []  # matrix of horizons
    for end in range(lookback, len(series) - max_horizon + 1):
        window = series[end - lookback:end].copy()
        target = series[end:end + max_horizon].copy()
        X.append(window)
        Ys.append(target)
    return np.asarray(X), np.asarray(Ys)


def _seasonal_naive_forecast(history: np.ndarray, horizon: int, period=30):
    # seasonal naive: use value from same seasonal lag; fallback to last value
    idx = -period
    if len(history) + idx < 0:
        return history[-1]
    return history[idx]


def run(seed: int = 42) -> Dict:
    np.random.seed(seed)
    series = _make_series(n=360, seed=seed)
    lookback = 30
    max_horizon = 7

    # Chronological split
    train_end = 300
    train_series = series[:train_end]
    test_series = series[train_end:]

    # Build windows on the training portion
    X_train, Y_train = _make_windows(train_series, lookback=lookback, max_horizon=max_horizon)

    # Fit direct models: one model per horizon
    direct_models = []
    for h in range(max_horizon):
        model = Ridge(alpha=1.0, random_state=seed)
        model.fit(X_train, Y_train[:, h])
        direct_models.append(model)

    # Fit one-step model for recursive strategy
    recursive_model = Ridge(alpha=1.0, random_state=seed)
    # targets are one-step-ahead from windows built on train_series shifted by 1
    X_train_one, Y_train_one = _make_windows(train_series, lookback=lookback, max_horizon=1)
    recursive_model.fit(X_train_one, Y_train_one[:, 0])

    # Backtest from fixed origins inside the test_series: choose origins every 7 days in the test range
    origins = []
    preds_direct = []  # list of arrays length max_horizon
    preds_recursive = []
    naive_preds = []
    truths = []

    test_start_global = train_end
    for offset in range(0, len(test_series) - max_horizon + 1, 7):
        origin_idx = test_start_global + offset
        origins.append(int(origin_idx))
        history = series[:origin_idx].copy()
        # direct
        x_in = history[-lookback:]
        direct_forecast = np.array([m.predict(x_in.reshape(1, -1))[0] for m in direct_models])
        preds_direct.append(direct_forecast)
        # recursive
        rec_forecast = []
        buffer = x_in.copy()
        for h in range(max_horizon):
            next_pred = recursive_model.predict(buffer.reshape(1, -1))[0]
            rec_forecast.append(next_pred)
            # roll buffer
            buffer = np.roll(buffer, -1)
            buffer[-1] = next_pred
        preds_recursive.append(np.array(rec_forecast))
        # seasonal naive baseline (use seasonal lag from history)
        naive_fore = np.array([_seasonal_naive_forecast(history, h + 1, period=30) for h in range(max_horizon)])
        naive_preds.append(naive_fore)
        # truth
        truth = series[origin_idx: origin_idx + max_horizon]
        truths.append(truth)

    preds_direct = np.vstack(preds_direct)
    preds_recursive = np.vstack(preds_recursive)
    naive_preds = np.vstack(naive_preds)
    truths = np.vstack(truths)

    # Compute horizon-wise MAE
    mae_direct = np.mean(np.abs(preds_direct - truths), axis=0)
    mae_recursive = np.mean(np.abs(preds_recursive - truths), axis=0)
    mae_naive = np.mean(np.abs(naive_preds - truths), axis=0)

    # Prepare a simple review-only report (string) listing issue dates and errors per horizon
    rows: List[str] = []
    hdr = [f"h{h+1}" for h in range(max_horizon)]
    rows.append("issue_idx," + ",".join(hdr))
    for i, idx in enumerate(origins):
        row = [str(idx)] + [f"{float(np.mean(np.abs(preds_direct[i, h] - truths[i, h]))):.3f}" for h in range(max_horizon)]
        rows.append(",".join(row))
    report = "\n".join(rows)

    # Return JSON-serialisable dict
    out = {
        "mae_direct_mean": float(mae_direct.mean()),
        "mae_recursive_mean": float(mae_recursive.mean()),
        "mae_naive_mean": float(mae_naive.mean()),
    }
    # add per-horizon
    for h in range(max_horizon):
        out[f"mae_direct_h{h+1}"] = float(mae_direct[h])
        out[f"mae_recursive_h{h+1}"] = float(mae_recursive[h])
        out[f"mae_naive_h{h+1}"] = float(mae_naive[h])
    # include review report and origins (as strings to keep JSON simple)
    out["review_report_csv"] = report
    out["issue_origin_indices"] = list(map(int, origins))
    out["note_model"] = "ridge_linear_models_direct_and_one_step_recursive"

    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
