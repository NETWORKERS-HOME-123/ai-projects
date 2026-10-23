import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def make_synthetic(seed: int = 42, n_items: int = 8, days: int = 210):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=days, freq="D")
    rows = []
    for item in range(n_items):
        base = rng.uniform(5, 30)
        weekly = 1 + 0.3 * np.sin(2 * np.pi * (dates.dayofweek) / 7 + rng.randn() * 0.2)
        trend = 1 + np.linspace(0, rng.uniform(-0.2, 0.8), days)
        promo = np.zeros(days)
        # occasional promotions
        for p in range(rng.randint(1, 4)):
            start = rng.randint(0, days - 14)
            promo[start : start + rng.randint(3, 10)] += rng.uniform(0.5, 1.5)
        # availability / stockouts with short runs
        avail = np.ones(days, dtype=int)
        for s in range(rng.randint(1, 4)):
            start = rng.randint(20, days - 10)
            length = rng.randint(1, 8)
            avail[start : start + length] = 0
        noise = rng.normal(scale=2.0, size=days)
        latent = base * weekly * trend * (1 + promo) + noise
        # observed sales cannot exceed availability: when unavailable, sales=0
        observed = np.where(avail == 1, np.maximum(0.0, latent).round(), 0.0)
        for i, d in enumerate(dates):
            rows.append(
                {
                    "item": f"sku_{item}",
                    "date": d,
                    "demand": float(observed[i]),
                    "avail": int(avail[i]),
                    "promo": float(promo[i] > 0),
                }
            )
    df = pd.DataFrame(rows)
    return df


def add_lags(df: pd.DataFrame, lags=(1, 7, 14)) -> pd.DataFrame:
    df = df.sort_values(["item", "date"]).copy()
    for l in lags:
        df[f"lag_{l}"] = df.groupby("item")["demand"].shift(l)
    df["dow"] = df["date"].dt.dayofweek
    return df


def seasonal_naive(series: np.ndarray, horizon: int, period: int = 7):
    # repeat the values from the last available period positions
    out = []
    for h in range(1, horizon + 1):
        idx = -period + ((h - 1) % period)
        out.append(series[idx])
    return np.array(out)


def intermittent_baseline(series: np.ndarray, horizon: int):
    pos = series[series > 0]
    if len(pos) == 0:
        return np.zeros(horizon)
    m = pos.mean()
    return np.full(horizon, m)


def run(seed: int = 42) -> Dict:
    # 1. Prepare data
    df = make_synthetic(seed=seed, n_items=8, days=210)
    df = add_lags(df)
    horizon = 30

    # choose an origin: we forecast the final `horizon` days from the state at origin = -horizon
    origin_date = df["date"].max() - pd.Timedelta(days=horizon)
    train_df = df[df["date"] <= origin_date].copy()
    test_df = df[df["date"] > origin_date].copy()

    features = ["lag_1", "lag_7", "lag_14", "dow", "promo", "avail"]
    # drop rows with NaN lags in training
    train_df = train_df.dropna(subset=["lag_1", "lag_7", "lag_14"]) 

    X_train = train_df[features]
    y_train = train_df["demand"]

    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    model.fit(X_train, y_train)

    # 2. Build the single-origin multi-step forecast (recursive): for each item, create the state at origin
    forecasts = []
    seasonal_naive_errors = []
    intermittent_errors = []
    maes = []
    biases = []
    stockout_maes = []

    for item, g in df.groupby("item"):
        hist = g[g["date"] <= origin_date].sort_values("date")
        # last observed series for seasonal naive and intermittent
        last_series = hist["demand"].values
        # prepare state for recursive prediction
        state = hist.iloc[-14:].copy()  # use last 14 to seed lags
        pred = []
        for h in range(1, horizon + 1):
            cur_date = origin_date + pd.Timedelta(days=h)
            dow = int(cur_date.dayofweek)
            # robust lag extraction relative to the end of state
            lag_1 = float(state["demand"].iloc[-1])
            lag_7 = float(state["demand"].iloc[-7]) if len(state) >= 7 else float(state["demand"].iloc[0])
            lag_14 = float(state["demand"].iloc[-14]) if len(state) >= 14 else float(state["demand"].iloc[0])
            # use last-known promo/avail as conservative guess rather than fixed constants
            promo = float(state["promo"].iloc[-1]) if "promo" in state.columns else 0.0
            avail = int(state["avail"].iloc[-1]) if "avail" in state.columns else 1
            x = np.array([[lag_1, lag_7, lag_14, dow, promo, avail]])
            # RandomForest will warn if feature names missing; passing numpy array is fine
            yhat = float(model.predict(x)[0])
            yhat = max(0.0, yhat)
            pred.append(yhat)
            # append predicted row to state so future lags use it; preserve promo/avail
            new_row = {"date": cur_date, "demand": yhat, "promo": promo, "avail": avail}
            state = pd.concat([state, pd.DataFrame([new_row])], ignore_index=True)
        # get actual future observed series for evaluation
        actual = df[(df["item"] == item) & (df["date"] > origin_date)]["demand"].values[:horizon]
        # baselines
        sn = seasonal_naive(last_series, horizon)
        ib = intermittent_baseline(last_series, horizon)
        # per-item metrics
        mae = float(np.mean(np.abs(np.array(pred) - actual)))
        bias = float(np.mean(np.array(pred) - actual))
        maes.append(mae)
        biases.append(bias)
        seasonal_naive_errors.append(float(np.mean(np.abs(sn - actual))))
        intermittent_errors.append(float(np.mean(np.abs(ib - actual))))
        # stockout breakdown: where actual sequence had avail==0 in test period
        stockout_mask = (
            df[(df["item"] == item) & (df["date"] > origin_date)]["avail"].values[:horizon] == 0
        )
        if stockout_mask.any():
            stock_mae = float(np.mean(np.abs(np.array(pred)[stockout_mask] - actual[stockout_mask])))
            stockout_maes.append(stock_mae)
        forecasts.append({"item": item, "forecast": [float(x) for x in pred]})

    result = {
        "mae_holdout": float(np.mean(maes)),
        "bias_holdout": float(np.mean(biases)),
        "seasonal_naive_mae": float(np.mean(seasonal_naive_errors)),
        "intermittent_mae": float(np.mean(intermittent_errors)),
        "stockout_mae": float(np.mean(stockout_maes)) if stockout_maes else 0.0,
        "horizon": horizon,
        "n_items": df["item"].nunique(),
        "forecasts_sample": forecasts[:3],
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
