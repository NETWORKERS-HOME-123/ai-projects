import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def make_synthetic_sales(n_days: int = 720, seed: int = 42):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    # Trend component
    trend = 0.05 * np.arange(n_days)
    # Weekly seasonality (7-day)
    dow = np.array([d.weekday() for d in dates])
    weekly = 10 * np.sin(2 * np.pi * dow / 7) + 5 * (dow >= 5)  # weekend lift
    # Promotion plan: known ahead (sparse binary promotions)
    promo = rng.binomial(1, 0.05, size=n_days)
    # Promotion effect applied to sales
    promo_effect = promo * (20 + rng.randn(n_days) * 5)
    # Noise
    noise = rng.randn(n_days) * 3
    sales = 50 + trend + weekly + promo_effect + noise
    sales = np.maximum(0.0, sales)
    df = pd.DataFrame({"date": dates, "sales": sales, "promo": promo})
    return df


def aggregate_horizon(df: pd.DataFrame, start_idx: int, h: int) -> float:
    # sum sales over next h days starting at start_idx+1
    return float(df.loc[start_idx + 1: start_idx + h, "sales"].sum())


def features_at_origin(df: pd.DataFrame, t: int, h: int) -> Dict:
    # features that would be available at time t to forecast next h days total
    vals = {}
    # recent lags from last observed day t
    for lag in (1, 7, 14, 28):
        idx = t - lag
        vals[f"lag_{lag}"] = float(df.loc[idx, "sales"]) if idx >= 0 else 0.0
    # weekday of origin
    vals["dow"] = int(df.loc[t, "date"].weekday())
    # known future promotions: sum of promos in the next h days (assumed planned and available)
    vals["promo_next_h"] = int(df.loc[t + 1: t + h, "promo"].sum())
    # rolling mean of last 28 days as baseline
    start = max(0, t - 28 + 1)
    vals["recent_mean_28"] = float(df.loc[start: t, "sales"].mean())
    return vals


def run(seed: int = 42) -> dict:
    rng = np.random.RandomState(seed)
    df = make_synthetic_sales(n_days=720, seed=seed)
    horizon = 7  # aggregate one-week ahead total

    # choose rolling origins: start after day 400 to ensure enough history
    origins = list(range(480, 640, 7))  # weekly origins
    records = []

    baseline_errors = []
    model_errors = []
    seasonal_failures = 0

    # We'll collect a small sample for reviewable output
    sample_rows: List[dict] = []

    for origin in origins:
        # prepare training data: times t where t + horizon <= origin
        train_indices = [t for t in range(28, origin - horizon + 1)]
        if len(train_indices) < 50:
            continue
        X_train = []
        y_train = []
        for t in train_indices:
            feats = features_at_origin(df, t, horizon)
            X_train.append(list(feats.values()))
            y_train.append(aggregate_horizon(df, t, horizon))
        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # fit a small RandomForest for deterministic behaviour
        model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=seed)
        model.fit(X_train, y_train)

        # prepare origin features (forecasting time)
        x_origin = np.array([list(features_at_origin(df, origin, horizon).values())])
        y_true = aggregate_horizon(df, origin, horizon)
        y_pred = float(model.predict(x_origin)[0])

        # seasonal-naive baseline: repeat last 7 days total
        last_7 = float(df.loc[origin - 7 + 1: origin, "sales"].sum())

        err_model = abs(y_true - y_pred)
        err_baseline = abs(y_true - last_7)
        model_errors.append(err_model)
        baseline_errors.append(err_baseline)

        # seasonal failure: compare deviation from recent baseline
        recent_baseline = float(df.loc[max(0, origin - 28 + 1): origin, "sales"].mean() * horizon)
        act_dev = y_true - recent_baseline
        pred_dev = y_pred - recent_baseline
        if (act_dev > 0 and pred_dev <= 0) or (act_dev < 0 and pred_dev >= 0):
            seasonal_failures += 1

        # store a small sample of values for review/plotting
        if len(sample_rows) < 6:
            sample_rows.append({
                "origin": str(df.loc[origin, "date"].date()),
                "y_true": float(y_true),
                "y_pred": float(y_pred),
                "baseline": float(last_7),
                "promo_next_h": int(df.loc[origin + 1: origin + horizon, "promo"].sum())
            })

    mae = float(np.mean(model_errors))
    baseline_mae = float(np.mean(baseline_errors))
    bias = float(np.mean([ (p - a) for p, a in zip([r for r in model_errors], [r for r in model_errors]) ]) )
    # Note: above bias computes mean absolute difference of errors; instead compute signed error properly
    signed_errors = []
    for idx, origin in enumerate(origins[: len(model_errors)]):
        # recompute for signed error
        # use sample predictions from earlier loop by recomputing simply
        pass
    # Simpler signed error calculation from records we have: compute model preds again deterministically
    signed_list = []
    model = RandomForestRegressor(n_estimators=50, max_depth=6, random_state=seed)
    # build global training on all data up to first origin to estimate bias conservatively
    # (bias is not the main metric; keep simple and deterministic)
    # Use all windows where t + horizon <= max(origins) - 1
    max_origin = max(origins)
    train_indices = [t for t in range(28, max_origin - horizon + 1)]
    X_all = []
    y_all = []
    for t in train_indices:
        feats = features_at_origin(df, t, horizon)
        X_all.append(list(feats.values()))
        y_all.append(aggregate_horizon(df, t, horizon))
    X_all = np.array(X_all)
    y_all = np.array(y_all)
    model.fit(X_all, y_all)
    signed_errors = []
    for origin in origins:
        x_origin = np.array([list(features_at_origin(df, origin, horizon).values())])
        y_true = aggregate_horizon(df, origin, horizon)
        y_pred = float(model.predict(x_origin)[0])
        signed_errors.append(y_pred - y_true)
    bias = float(np.mean(signed_errors))

    seasonal_failure_rate = seasonal_failures / max(1, len(origins))

    # main metric: neg_mae so higher is better for test floors
    neg_mae = -mae

    result = {
        "model": "RandomForestRegressor",
        "horizon_days": horizon,
        "origins_evaluated": int(len(origins)),
        "mae": mae,
        "baseline_mae": baseline_mae,
        "neg_mae": neg_mae,
        "bias": bias,
        "seasonal_failure_rate": seasonal_failure_rate,
        "sample_forecasts": sample_rows,
    }
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
