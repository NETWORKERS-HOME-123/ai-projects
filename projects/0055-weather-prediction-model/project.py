import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error, r2_score


def make_station_data(n_days: int = 1200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2018-01-01", periods=n_days, freq="D")
    doy = dates.dayofyear.values
    # Base seasonal temperature (annual sinusoid) + trend + noise
    seasonal = 10.0 + 10.0 * np.sin(2 * np.pi * (doy / 365.25))
    trend = 0.002 * np.arange(n_days)  # slight warming trend for realism
    noise = rng.normal(scale=2.0, size=n_days)
    max_temp = seasonal + trend + noise
    # Create plausible predictors available at issue time (previous day observations)
    prev_temp = np.roll(max_temp, 1)
    prev_temp[0] = max_temp[0]  # first day fallback
    humidity = 60 + 20 * np.sin(2 * np.pi * (doy / 365.25 + 0.3)) + rng.normal(scale=5.0, size=n_days)
    wind_speed = np.abs(rng.normal(loc=4.0, scale=1.5, size=n_days))
    df = pd.DataFrame({
        "date": dates,
        "doy": doy,
        "prev_temp": prev_temp,
        "prev_humidity": humidity,
        "prev_wind": wind_speed,
        "max_temp_tgt": max_temp,
    })
    return df


def chronological_split(df: pd.DataFrame, test_fraction: float = 0.2):
    n = len(df)
    split = int(n * (1 - test_fraction))
    train = df.iloc[:split].reset_index(drop=True)
    test = df.iloc[split:].reset_index(drop=True)
    return train, test


def persistence_forecast(test: pd.DataFrame) -> np.ndarray:
    # Persistence: predict today's max temp = yesterday's observed max temp
    return test["prev_temp"].values


def climatology_forecast(train: pd.DataFrame, test: pd.DataFrame) -> np.ndarray:
    # Seasonal climatology: mean max_temp for each day-of-year in train
    climatology = train.groupby("doy")["max_temp_tgt"].mean()
    return test["doy"].map(climatology).fillna(train["max_temp_tgt"].mean()).values


def fit_model(train: pd.DataFrame, seed: int = 42):
    X = train[["prev_temp", "prev_humidity", "prev_wind", "doy"]].values
    y = train["max_temp_tgt"].values
    model = make_pipeline(StandardScaler(), LinearRegression())
    # LinearRegression is deterministic; seed parameter left for API consistency
    model.fit(X, y)
    return model


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"mae": mae, "r2": r2}


def run(seed: int = 42) -> Dict:
    # Build synthetic station data (daily obs) as a faithful small stand-in
    df = make_station_data(n_days=1200, seed=seed)
    train, test = chronological_split(df, test_fraction=0.2)

    model = fit_model(train, seed=seed)
    X_test = test[["prev_temp", "prev_humidity", "prev_wind", "doy"]].values
    y_test = test["max_temp_tgt"].values
    y_pred = model.predict(X_test)

    # Baselines
    y_persist = persistence_forecast(test)
    y_clim = climatology_forecast(train, test)

    # Evaluate overall
    model_metrics = evaluate(y_test, y_pred)
    persist_metrics = evaluate(y_test, y_persist)
    clim_metrics = evaluate(y_test, y_clim)

    # Evaluate on extremes (top and bottom 5% by true temperature)
    n = len(y_test)
    k = max(1, int(0.05 * n))
    order = np.argsort(y_test)
    low_idx = order[:k]
    high_idx = order[-k:]
    extremes = {}
    extremes["low_mae_model"] = float(mean_absolute_error(y_test[low_idx], y_pred[low_idx]))
    extremes["high_mae_model"] = float(mean_absolute_error(y_test[high_idx], y_pred[high_idx]))

    out = {
        "horizon_hours": 24,
        "model": "LinearRegression",
        "n_samples": int(len(df)),
        "test_size": int(len(test)),
        "mae": model_metrics["mae"],
        "r2": model_metrics["r2"],
        "persistence_mae": persist_metrics["mae"],
        "climatology_mae": clim_metrics["mae"],
        "extreme_low_mae": extremes["low_mae_model"],
        "extreme_high_mae": extremes["high_mae_model"],
        "notes": (
            "Synthetic station-like data; baselines: persistence (yesterday) and seasonal climatology. "
            "See official warnings from national services (e.g. https://mausam.imd.gov.in/) for real decisions."
        ),
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
