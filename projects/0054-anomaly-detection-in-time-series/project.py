import json
from typing import List, Tuple, Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def generate_series(seed: int = 42) -> Tuple[pd.DataFrame, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    rng = np.random.RandomState(seed)
    periods = 180 * 24
    idx = pd.date_range("2021-01-01", periods=periods, freq="h")
    t = np.arange(periods)
    trend = 0.0005 * t
    daily = 2.0 * np.sin(2 * np.pi * (t % 24) / 24)
    noise = rng.normal(scale=0.5, size=periods)
    base = 10.0 + trend + daily + noise

    missing = np.zeros(periods, dtype=bool)
    for a, b in [(200, 260), (1500, 1524)]:
        missing[a:b] = True

    series = base.copy()
    series[missing] = np.nan

    events: List[Tuple[pd.Timestamp, pd.Timestamp]] = []
    events.append((idx[3000], idx[3004]))
    series[3000:3005] += 6.0
    events.append((idx[3400], idx[3430]))
    series[3400:3431] += 3.5
    events.append((idx[3900], idx[3948]))
    series[3900:3949] += 2.5 * np.sin(2 * np.pi * (np.arange(49) % 24) / 24 + 1.0)

    df = pd.DataFrame({"timestamp": idx, "value": series})
    return df, events


def design_matrix(timestamps: pd.Series) -> np.ndarray:
    ts = pd.Series(pd.to_datetime(timestamps))
    hours = ((ts - ts.min()).dt.total_seconds() / 3600.0).to_numpy()
    sin24 = np.sin(2 * np.pi * (hours % 24) / 24)
    cos24 = np.cos(2 * np.pi * (hours % 24) / 24)
    return np.column_stack([hours, sin24, cos24])


def train_baseline(df_train: pd.DataFrame) -> LinearRegression:
    X = design_matrix(df_train["timestamp"])
    y = df_train["value"].values
    mask = ~np.isnan(y)
    model = LinearRegression()
    model.fit(X[mask], y[mask])
    return model


def compute_threshold(train_residuals: np.ndarray, alpha: float = 0.975) -> float:
    thr = np.percentile(np.abs(train_residuals), 100.0 * alpha)
    return float(thr * 1.05 + 1e-8)


def rolling_forecast_single_origin(model: LinearRegression, df_test: pd.DataFrame) -> np.ndarray:
    X_test = design_matrix(df_test["timestamp"])
    return model.predict(X_test)


def detect_alerts(timestamps: pd.Series, residuals: np.ndarray, threshold: float) -> List[Tuple[pd.Timestamp, pd.Timestamp, int]]:
    above = np.abs(residuals) > threshold
    alerts: List[Tuple[pd.Timestamp, pd.Timestamp, int]] = []
    i = 0
    n = len(above)
    while i < n:
        if above[i]:
            j = i
            while j + 1 < n and above[j + 1]:
                j += 1
            window_idx = np.arange(i, j + 1)
            peak = window_idx[np.argmax(np.abs(residuals[window_idx]))]
            alerts.append((timestamps.iloc[i], timestamps.iloc[j], int(peak)))
            i = j + 1
        else:
            i += 1
    return alerts


def evaluate_alerts(alerts: List[Tuple[pd.Timestamp, pd.Timestamp, int]], annotated: List[Tuple[pd.Timestamp, pd.Timestamp]], test_start: pd.Timestamp, test_end: pd.Timestamp) -> Dict[str, float]:
    hits = 0
    delays = []
    considered = 0
    for a_start, a_end in annotated:
        if a_end < test_start or a_start > test_end:
            continue
        considered += 1
        found = False
        first_alert_time = None
        for s, e, _ in alerts:
            if s <= a_end and e >= a_start:
                found = True
                first_alert_time = s
                break
        if found:
            hits += 1
            delay = (first_alert_time - a_start).total_seconds() / 3600.0
            delays.append(max(0.0, delay))
    event_recall = hits / considered if considered > 0 else 0.0
    false_count = 0
    for s, e, _ in alerts:
        overlap = any(not (e < a_start or s > a_end) for a_start, a_end in annotated)
        if not overlap:
            false_count += 1
    days = (test_end - test_start).total_seconds() / (3600.0 * 24.0)
    false_per_day = false_count / days if days > 0 else float(false_count)
    mean_delay = float(np.mean(delays)) if len(delays) > 0 else float("nan")
    return {
        "event_recall": float(event_recall),
        "false_alerts_per_day": float(false_per_day),
        "mean_detection_delay_hours": mean_delay,
        "alerts_count": int(len(alerts)),
        "annotated_events_considered": int(considered),
    }


def make_review_timeline(alerts: List[Tuple[pd.Timestamp, pd.Timestamp, int]], df_test: pd.DataFrame) -> List[Dict[str, str]]:
    out = []
    for s, e, peak_idx in alerts[:8]:
        context_slice = df_test[(df_test["timestamp"] >= s - pd.Timedelta(hours=6)) & (df_test["timestamp"] <= e + pd.Timedelta(hours=6))]
        peak_value = float(context_slice.loc[context_slice.index == peak_idx, "value"].values[0]) if peak_idx in context_slice.index else float("nan")
        out.append({"alert_start": str(s), "alert_end": str(e), "peak_value": float(peak_value), "review_note": "unreviewed"})
    return out


def run(seed: int = 42) -> Dict:
    df, annotated = generate_series(seed=seed)
    split_idx = int(120 * 24)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    model = train_baseline(df_train)
    X_train = design_matrix(df_train["timestamp"])
    preds_train = model.predict(X_train)
    train_mask = ~np.isnan(df_train["value"].values)
    train_residuals = df_train["value"].values[train_mask] - preds_train[train_mask]
    threshold = compute_threshold(train_residuals, alpha=0.975)

    preds_test = rolling_forecast_single_origin(model, df_test)
    residuals = np.full(len(df_test), np.nan)
    obs_mask = ~np.isnan(df_test["value"].values)
    residuals[obs_mask] = df_test["value"].values[obs_mask] - preds_test[obs_mask]
    alerts = detect_alerts(df_test["timestamp"], np.nan_to_num(residuals, nan=0.0), threshold)

    metrics = evaluate_alerts(alerts, annotated, df_test["timestamp"].iloc[0], df_test["timestamp"].iloc[-1])
    review_timeline = make_review_timeline(alerts, df_test)
    return {
        "event_recall": metrics["event_recall"],
        "false_alerts_per_day": metrics["false_alerts_per_day"],
        "mean_detection_delay_hours": metrics["mean_detection_delay_hours"],
        "alerts_count": metrics["alerts_count"],
        "annotated_events_considered": metrics["annotated_events_considered"],
        "threshold_used": float(threshold),
        "review_timeline_preview": review_timeline,
    }


if __name__ == "__main__":
    print(json.dumps(run(42), indent=2))
