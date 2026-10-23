import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score


def make_synthetic_hourly(seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    hours = 24
    days = 365 + 14  # one year plus two weeks for holdout
    total_hours = days * hours
    start = pd.Timestamp("2020-01-01")
    # use lowercase 'h' for hourly frequency to be compatible with pandas versions
    idx = pd.date_range(start, periods=total_hours, freq="h")
    df = pd.DataFrame(index=idx)
    df["hour"] = df.index.hour
    df["dow"] = df.index.dayofweek
    # simple holiday mask: treat fixed set of dates as holidays
    holidays = set(pd.to_datetime(["2020-01-26", "2020-08-15", "2020-10-25", "2020-12-25"]))
    df["is_holiday"] = df.index.normalize().isin(holidays).astype(int)
    # base load: daily and weekly patterns
    base = 50 + 10 * np.sin(2 * np.pi * df["hour"] / 24) + 5 * ((df["dow"] >= 5).astype(float))
    # temperature (actual) follows seasonal + daily
    day_of_year = df.index.dayofyear
    actual_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year / 365)) + 5 * np.sin(2 * np.pi * df["hour"] / 24)
    # as-issued forecast: actual plus a small systematic bias and noise (this simulates forecast available at production time)
    forecast_temp = actual_temp + rng.normal(loc=0.5, scale=1.0, size=len(actual_temp))
    # consumption depends on base, temperature and holiday effect
    consumption = base - 0.5 * actual_temp + 5 * df["is_holiday"] + rng.normal(scale=2.0, size=len(actual_temp))
    df["actual_temp"] = actual_temp
    df["forecast_temp"] = forecast_temp
    df["consumption"] = consumption
    # Introduce a few missing meter readings (to trigger warnings)
    missing_idx = rng.choice(len(df), size=5, replace=False)
    df.iloc[missing_idx, df.columns.get_loc("consumption")] = np.nan
    # Introduce a few stale forecast flags by setting a last_update time far in the past; store as boolean
    stale_idx = rng.choice(len(df), size=8, replace=False)
    df["forecast_stale"] = False
    df.iloc[stale_idx, df.columns.get_loc("forecast_stale")] = True
    return df


def create_lag_features(df: pd.DataFrame, lags=(1, 24, 24 * 7)) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out["consumption"].shift(lag)
    # Use only forecast_temp (as-issued) not actual_temp to avoid future leakage
    out["temp"] = out["forecast_temp"]
    # Drop rows where all lags are NaN (start of series)
    return out


def train_and_evaluate(df: pd.DataFrame, seed: int = 42) -> Dict[str, Any]:
    # Chronological split: train on initial part, test on last 14 days
    hours = 24
    holdout_hours = 14 * hours
    df_features = create_lag_features(df)
    train = df_features.iloc[: -holdout_hours].dropna(subset=["lag_1"])  # ensure lag_1 present
    test = df_features.iloc[-holdout_hours :].copy()

    # Baseline: same-hour-last-week (lag_168)
    # use a writable copy to allow filling fallbacks
    baseline_pred = test["lag_168"].values.copy()
    # For rows where lag_168 is NaN, fallback to last hour or mean
    nan_mask = np.isnan(baseline_pred)
    if nan_mask.any():
        fallback = test["lag_24"].fillna(df["consumption"].mean()).values
        baseline_pred[nan_mask] = fallback[nan_mask]

    # Prepare training matrices (drop rows with NaN target)
    train = train.dropna(subset=["consumption"])
    X_train = train[["lag_1", "lag_24", "lag_168", "hour", "dow", "is_holiday", "temp"]].fillna(0).values
    y_train = train["consumption"].values

    # For test, we must not use future true consumption; use only features present (lags + forecast temp)
    X_test = test[["lag_1", "lag_24", "lag_168", "hour", "dow", "is_holiday", "temp"]].fillna(0).values
    # Replace NaN in lags with zeros (simulating missing meter fallback)

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Where consumption was missing in test, we cannot compute error for that hour; keep mask
    valid_mask = ~test["consumption"].isna().values
    truth = test["consumption"].values[valid_mask]
    preds_valid = preds[valid_mask]
    baseline_valid = baseline_pred[valid_mask]

    mae = float(mean_absolute_error(truth, preds_valid))
    r2 = float(r2_score(truth, preds_valid))
    baseline_mae = float(mean_absolute_error(truth, baseline_valid))

    # Peak period MAE: morning 6-9 and evening 17-20
    test_hours = test.index.hour.values[valid_mask]
    peak_mask = ((test_hours >= 6) & (test_hours <= 9)) | ((test_hours >= 17) & (test_hours <= 20))
    if peak_mask.any():
        peak_mae = float(mean_absolute_error(truth[peak_mask], preds_valid[peak_mask]))
    else:
        peak_mae = float('nan')

    # Holiday MAE
    holiday_mask = test["is_holiday"].values[valid_mask].astype(bool)
    if holiday_mask.any():
        holiday_mae = float(mean_absolute_error(truth[holiday_mask], preds_valid[holiday_mask]))
    else:
        holiday_mae = float('nan')

    # Warnings
    missing_meter_warning = int(test["consumption"].isna().any())
    stale_input_warning = int(test["forecast_stale"].any())

    # Simple textual read-only load planning chart (ASCII sparkline of truth vs prediction aggregated daily)
    daily = pd.DataFrame({"truth": test["consumption"], "pred": preds}, index=test.index)
    daily = daily.resample("D").mean()
    def sparkline(series):
        vals = series.fillna(series.mean()).values
        mins, maxs = vals.min(), vals.max()
        if maxs - mins < 1e-6:
            return "[no-variation]"
        chars = "▁▂▃▄▅▆▇█"
        scaled = ((vals - mins) / (maxs - mins) * (len(chars) - 1)).round().astype(int)
        return "".join(chars[i] for i in scaled)
    chart = {
        "daily_truth_sparkline": sparkline(daily["truth"]),
        "daily_pred_sparkline": sparkline(daily["pred"]),
    }

    return {
        "r2": r2,
        "mae": mae,
        "baseline_mae": baseline_mae,
        "peak_mae": peak_mae,
        "holiday_mae": holiday_mae,
        "missing_meter_warning": missing_meter_warning,
        "stale_input_warning": stale_input_warning,
        "load_planning_chart": chart,
    }


def run(seed: int = 42) -> Dict[str, Any]:
    df = make_synthetic_hourly(seed=seed)
    results = train_and_evaluate(df, seed=seed)
    # Keep only JSON-serialisable simple types (chart contains strings)
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
