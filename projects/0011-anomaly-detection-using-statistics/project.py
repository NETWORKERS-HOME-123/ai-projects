import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def run(seed: int = 42) -> Dict:
    """Generate a synthetic seasonal sensor, build baseline thresholds,
    flag anomalies on a held-out segment, and report basic metrics.

    Returns a JSON-serialisable dict with precision/recall and counts.
    """
    rng = np.random.RandomState(seed)

    # Synthesize time series: seasonal + noise
    n = 800
    t = np.arange(n)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 100)  # slow seasonality
    noise = rng.normal(scale=0.8, size=n)
    signal = seasonal + noise

    # Inject anomalies (ground truth): spikes and sustained drift
    labels = np.zeros(n, dtype=int)
    anomaly_indices = [250, 251, 252, 400, 401, 402, 650]  # spikes / short bursts
    labels[anomaly_indices] = 1
    # sustained positive drift window
    labels[480:500] = 1
    signal_with_anom = signal.copy()
    signal_with_anom[anomaly_indices] += rng.uniform(6.0, 9.0, size=len(anomaly_indices))
    signal_with_anom[480:500] += 4.5  # sustained shift

    # Pack into DataFrame for convenience
    df = pd.DataFrame({"t": t, "x": signal_with_anom, "anomaly": labels})

    # Split into baseline (calibration) and held-out evaluation segments
    # We choose a contiguous baseline window at the start to emulate initial calibration
    # Adjust baseline to exclude early injected spikes so calibration is not contaminated
    baseline_end = 200
    baseline = df.loc[df.t < baseline_end].copy()
    held_out = df.loc[df.t >= baseline_end].copy().reset_index(drop=True)

    # 1) Classic z-score thresholds from baseline
    mu = baseline.x.mean()
    sigma = baseline.x.std(ddof=0)
    # Use a slightly tighter z threshold and focus on upper tail because injected
    # anomalies are positive spikes/drift
    k_z = 2.5
    upper_z = mu + k_z * sigma
    lower_z = mu - k_z * sigma

    # 2) Robust median/MAD threshold from baseline
    med = baseline.x.median()
    mad = np.median(np.abs(baseline.x - med))
    # Scale MAD to be comparable to std for normal data (approx factor 1.4826)
    mad_scaled = mad * 1.4826 if mad > 0 else 0.0
    # Slightly tighter MAD multiplier as well
    k_mad = 3.0
    upper_mad = med + k_mad * mad_scaled
    lower_mad = med - k_mad * mad_scaled

    # Flagging rule on held-out: flag if either rule signals.
    # Since anomalies are injected as positive shifts, only check upper thresholds to
    # avoid flagging symmetric natural variability on the lower side.
    def flag_row(val: float) -> bool:
        z_flag = (val > upper_z)
        mad_flag = (val > upper_mad)
        return bool(z_flag or mad_flag)

    held_out["flag"] = held_out.x.apply(flag_row).astype(int)

    # Validation against ground truth: count TP/FP/FN
    tp = int(((held_out.flag == 1) & (held_out.anomaly == 1)).sum())
    fp = int(((held_out.flag == 1) & (held_out.anomaly == 0)).sum())
    fn = int(((held_out.flag == 0) & (held_out.anomaly == 1)).sum())

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

    # Return numeric thresholds and counts so tests can inspect them
    result = {
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "baseline_mean": float(mu),
        "baseline_std": float(sigma),
        "baseline_median": float(med),
        "baseline_mad_scaled": float(mad_scaled),
        "upper_z": float(upper_z),
        "lower_z": float(lower_z),
        "upper_mad": float(upper_mad),
        "lower_mad": float(lower_mad),
        "held_out_size": int(len(held_out)),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
