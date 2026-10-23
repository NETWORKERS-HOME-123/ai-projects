import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def _make_synthetic_traffic(n_segments=10, hours_per_segment=200, seed=42):
    rng = np.random.RandomState(seed)
    rows = []
    # pandas may accept lowercase 'h' for hourly frequency across versions
    base_times = pd.date_range("2021-01-01", periods=hours_per_segment, freq="h")
    for seg in range(n_segments):
        amp = 50 + rng.rand() * 100
        offset = rng.rand() * 2 * np.pi
        noise_scale = 5 + rng.rand() * 10
        trend = 0.01 * np.arange(hours_per_segment) * (1 + 0.1 * rng.randn())
        for t_idx, ts in enumerate(base_times):
            hour = ts.hour
            diurnal = amp * (0.6 * np.sin((hour / 24) * 2 * np.pi + offset) + 0.4 * np.cos((hour / 24) * 4 * np.pi + offset / 2))
            count = max(0.0, 20 + diurnal + trend[t_idx] + rng.randn() * noise_scale)
            rows.append((seg, ts, int(ts.hour), int(ts.weekday() >= 5), count))
    df = pd.DataFrame(rows, columns=["segment", "ts", "hour", "is_weekend", "count"])
    rng = np.random.RandomState(seed + 1)
    n_incidents = max(1, n_segments // 2)
    incident_rows = np.zeros(len(df), dtype=bool)
    for _ in range(n_incidents):
        seg = int(rng.randint(0, n_segments))
        start = int(rng.randint(0, hours_per_segment - 5))
        duration = int(rng.randint(1, 6))
        mask = (df.segment == seg) & (df.ts >= df.ts.iloc[start]) & (df.ts < df.ts.iloc[start + duration])
        incident_rows |= mask.values
    df["incident"] = incident_rows.astype(int)
    df.loc[df["incident"] == 1, "count"] *= 0.6
    df["count"] = df["count"].round(1)
    return df


def _make_features(df: pd.DataFrame, lags=(1, 24)) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values(["segment", "ts"]).reset_index(drop=True)
    for lag in lags:
        data[f"lag_{lag}"] = data.groupby("segment")["count"].shift(lag)
    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data = data.dropna().reset_index(drop=True)
    return data


def run(seed: int = 42) -> Dict:
    df = _make_synthetic_traffic(n_segments=10, hours_per_segment=200, seed=seed)
    rng = np.random.RandomState(seed)
    segments = sorted(df.segment.unique())
    heldout_segments = list(rng.choice(segments, size=2, replace=False))
    train_segments = [s for s in segments if s not in heldout_segments]
    feat = _make_features(df, lags=(1, 24))

    # compute per-segment cutoff without using groupby.apply which can drop columns
    cutoff = feat.groupby("segment")["ts"].transform(lambda s: s.min() + pd.Timedelta(hours=150))
    feat["is_train"] = feat["ts"] < cutoff

    train = feat[(feat["segment"].isin(train_segments)) & (feat["is_train"])]
    test_future = feat[(feat["segment"].isin(train_segments)) & (~feat["is_train"])]
    held_segments = feat[(feat["segment"].isin(heldout_segments)) & (~feat["is_train"])]
    feature_cols = ["lag_1", "lag_24", "hour_sin", "hour_cos", "is_weekend", "incident"]
    X_train = train[feature_cols].values
    y_train = train["count"].values
    X_test_future = test_future[feature_cols].values
    y_test_future = test_future["count"].values
    X_held = held_segments[feature_cols].values
    y_held = held_segments["count"].values

    model = RandomForestRegressor(n_estimators=100, random_state=seed, n_jobs=1)
    model.fit(X_train, y_train)

    def predict_persistence(df_rows):
        return df_rows["lag_1"].values

    y_pred_future = model.predict(X_test_future)
    y_pred_future_pers = predict_persistence(test_future)
    y_pred_held = model.predict(X_held)
    y_pred_held_pers = predict_persistence(held_segments)

    def rmse(a, b):
        return float(np.sqrt(mean_squared_error(a, b)))

    def eval_set(y_true, y_pred, meta_df, prefix):
        overall_rmse = rmse(y_true, y_pred)
        overall_r2 = float(r2_score(y_true, y_pred))
        rush_mask = meta_df["hour"].isin([7, 8, 9, 17, 18, 19]).values
        rush_rmse = rmse(y_true[rush_mask], y_pred[rush_mask]) if rush_mask.sum() > 0 else float("nan")
        inc_mask = meta_df["incident"].astype(bool).values
        inc_rmse = rmse(y_true[inc_mask], y_pred[inc_mask]) if inc_mask.sum() > 0 else float("nan")
        return {f"{prefix}_rmse": overall_rmse, f"{prefix}_r2": overall_r2, f"{prefix}_rush_rmse": rush_rmse, f"{prefix}_incident_rmse": inc_rmse}

    test_future_meta = test_future.reset_index(drop=True)
    held_meta = held_segments.reset_index(drop=True)
    results = {}
    results.update(eval_set(y_test_future, y_pred_future, test_future_meta, "future"))
    results.update(eval_set(y_test_future, y_pred_future_pers, test_future_meta, "future_persistence"))
    results.update(eval_set(y_held, y_pred_held, held_meta, "held_segments"))
    results.update(eval_set(y_held, y_pred_held_pers, held_meta, "held_segments_persistence"))

    last_obs = df.groupby("segment")["ts"].max()
    max_age_hours = (pd.Timestamp(df.ts.max()) - last_obs).dt.total_seconds() / 3600.0
    stale_segments = list(max_age_hours[max_age_hours > 3].index)
    planning = {"heldout_segments": heldout_segments, "stale_segments": stale_segments, "note": "Estimates are for planning only. Do not trigger automatic control actions."}

    output = {
        "held_segments_r2": results["held_segments_r2"],
        "held_segments_rmse": results["held_segments_rmse"],
        "held_segments_rush_rmse": results["held_segments_rush_rmse"],
        "held_segments_incident_rmse": results["held_segments_incident_rmse"],
        "future_r2": results["future_r2"],
        "future_rmse": results["future_rmse"],
        "future_persistence_rmse": results["future_persistence_rmse"],
        "planning": planning,
    }
    return output


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
