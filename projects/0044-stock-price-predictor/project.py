import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


def generate_gbm_prices(n_days: int, s0: float, mu: float, sigma: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    dt = 1.0
    shocks = rng.normal(loc=(mu - 0.5 * sigma ** 2) * dt, scale=sigma * np.sqrt(dt), size=n_days)
    log_price = np.log(s0) + np.cumsum(shocks)
    return np.exp(log_price)


def build_features(prices: np.ndarray, lags=(1, 2, 5), ma_windows=(3, 5)) -> pd.DataFrame:
    df = pd.DataFrame({"price": prices})
    for lag in lags:
        df[f"lag_{lag}"] = df["price"].shift(lag)
    for w in ma_windows:
        df[f"ma_{w}"] = df["price"].rolling(window=w).mean()
    # returns
    df["ret_1"] = df["price"].pct_change(1)
    df["ret_5"] = df["price"].pct_change(5)
    return df


def make_targets(df: pd.DataFrame, horizons=(1, 5, 10)) -> pd.DataFrame:
    for h in horizons:
        df[f"target_h{h}"] = df["price"].shift(-h)
    return df


def walk_forward_evaluate(X: pd.DataFrame, y: pd.Series, dates: pd.Index, model, start_train_size: int, step: int) -> Dict[str, float]:
    preds = []
    truths = []
    n = len(X)
    train_start = 0
    train_end = start_train_size
    while train_end + step <= n:
        X_train = X.iloc[train_start:train_end]
        y_train = y.iloc[train_start:train_end]
        X_test = X.iloc[train_end: train_end + step]
        y_test = y.iloc[train_end: train_end + step]
        if X_train.shape[0] < 10:
            break
        model.fit(X_train, y_train)
        p = model.predict(X_test)
        preds.append(p)
        truths.append(y_test.values)
        train_end += step
    if len(preds) == 0:
        return {"mae": float("nan"), "r2": float("nan")}
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return {"mae": float(mean_absolute_error(truths, preds)), "r2": float(r2_score(truths, preds))}


def run(seed: int = 42) -> dict:
    # Configuration
    np.random.seed(seed)
    n_days = 500
    s0 = 100.0
    mu = 0.0005  # daily drift
    sigma = 0.01

    # 1) Prepare synthetic adjusted-price series (small stand-in for licensed data)
    prices = generate_gbm_prices(n_days=n_days, s0=s0, mu=mu, sigma=sigma, seed=seed)
    df = build_features(prices)
    df = make_targets(df)
    df = df.dropna().reset_index(drop=True)

    # features usable at prediction time
    feature_cols = [c for c in df.columns if c.startswith("lag_") or c.startswith("ma_") or c.startswith("ret_")]

    results = {}
    horizons = [1, 5, 10]

    # Walk-forward parameters
    start_train_size = 200
    step = 20
    model = Ridge(alpha=1.0, random_state=seed)

    for h in horizons:
        X = df[feature_cols].copy()
        y = df[f"target_h{h}"].copy()
        # Align: drop rows where target is nan (already removed) and any remaining nan in features
        valid = X.notnull().all(axis=1) & y.notnull()
        Xv = X[valid].reset_index(drop=True)
        yv = y[valid].reset_index(drop=True)
        dates = Xv.index
        res = walk_forward_evaluate(Xv, yv, dates, model, start_train_size=start_train_size, step=step)
        results[f"mae_horizon_{h}"] = res["mae"]
        results[f"r2_horizon_{h}"] = res["r2"]

        # Baseline 1: last-price baseline (forecast = last observed price)
        baseline_preds = Xv["lag_1"].values
        baseline_mae = mean_absolute_error(yv.values, baseline_preds)
        baseline_r2 = r2_score(yv.values, baseline_preds)
        results[f"baseline_last_r2_h{h}"] = float(baseline_r2)
        results[f"baseline_last_mae_h{h}"] = float(baseline_mae)

        # Baseline 2: simple-return extrapolation: apply recent mean return over horizon
        recent_ret = Xv["ret_1"].fillna(0).rolling(window=5, min_periods=1).mean().values
        baseline2_preds = Xv["lag_1"].values * (1.0 + recent_ret * h)
        baseline2_mae = mean_absolute_error(yv.values, baseline2_preds)
        baseline2_r2 = r2_score(yv.values, baseline2_preds)
        results[f"baseline_return_r2_h{h}"] = float(baseline2_r2)
        results[f"baseline_return_mae_h{h}"] = float(baseline2_mae)

    # Include a simple illustrative transaction cost metric (not a proper backtest):
    # cost_per_trade is expressed as price units per round-trip; compute average cost assuming one trade per prediction
    cost_per_trade = 0.05
    results["avg_transaction_cost_per_prediction"] = float(cost_per_trade)

    # Educational note returned for display (brief)
    results["note"] = "Synthetic GBM used. Model: Ridge regression. Chronological walk-forward evaluation. Educational only."

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
