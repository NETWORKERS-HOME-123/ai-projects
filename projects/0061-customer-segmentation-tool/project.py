import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score


def _make_rfm(seed: int, n_per_segment: int = 150, centers: np.ndarray | None = None, drift: float = 0.0):
    rng = np.random.RandomState(seed)
    if centers is None:
        centers = np.array([
            [10.0, 12.0, 300.0],   # recent, frequent, high spend
            [60.0, 4.0, 80.0],     # less recent, moderate freq, mid spend
            [200.0, 1.0, 20.0],    # very old, rare, low spend
        ])
    data = []
    for i, c in enumerate(centers):
        # add gaussian jitter; drift shifts centers slightly
        shift = drift * (i - 1)
        cov = np.diag([10.0, 2.0, 50.0])
        samples = rng.multivariate_normal(c + shift, cov, size=n_per_segment)
        data.append(samples)
    X = np.vstack(data)
    # clip to sensible values
    X[:, 0] = np.clip(X[:, 0], 0.0, None)  # recency >= 0
    X[:, 1] = np.clip(X[:, 1], 0.0, None)  # frequency >= 0
    X[:, 2] = np.clip(X[:, 2], 0.0, None)  # spend >= 0
    cols = ["recency", "frequency", "spend"]
    return pd.DataFrame(X, columns=cols)


def run(seed: int = 42) -> Dict:
    """Run the segmentation experiment and return metrics and short summaries.

    Returns a JSON-serialisable dict with keys:
      - best_k: chosen cluster count
      - silhouette: silhouette score on training period
      - stability_ari: adjusted rand index comparing labels across two periods
      - n_segments: number of segments in the chosen model
    """
    rng = np.random.RandomState(seed)

    # Prepare two periods (period A: training, period B: later reporting period)
    period_a = _make_rfm(seed=seed, n_per_segment=150, drift=0.0)
    period_b = _make_rfm(seed=seed + 1, n_per_segment=150, drift=5.0)  # small drift

    # Build experiment: scale numeric features
    scaler = StandardScaler()
    Xa = scaler.fit_transform(period_a.values)
    Xb = scaler.transform(period_b.values)

    # Compare K choices and pick the best by silhouette on period A
    best_k = None
    best_score = -1.0
    best_labels = None
    best_model = None
    for k in [2, 3, 4, 5]:
        model = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = model.fit_predict(Xa)
        # silhouette requires at least 2 clusters and fewer than n_samples
        score = silhouette_score(Xa, labels) if 1 < k < len(Xa) else -1.0
        if score > best_score:
            best_score = float(score)
            best_k = int(k)
            best_labels = labels.copy()
            best_model = model

    # Assign clusters to period B by predicting nearest center
    labels_b = best_model.predict(Xb)

    # Evaluate stability across periods using Adjusted Rand Index
    ari = float(adjusted_rand_score(best_labels, labels_b))

    # Produce simple neutral summaries for reporting (aggregate means and sizes)
    df_a = pd.DataFrame(period_a.copy())
    df_a["cluster"] = best_labels
    summary = (
        df_a.groupby("cluster").agg(
            size=("recency", "count"),
            recency_mean=("recency", "mean"),
            frequency_mean=("frequency", "mean"),
            spend_mean=("spend", "mean"),
        )
        .reset_index()
        .sort_values("cluster")
    )

    # Prepare a concise output (only JSON-serialisable types)
    out = {
        "best_k": best_k,
        "n_segments": int(summary["size"].sum()),
        "silhouette": round(float(best_score), 4),
        "stability_ari": round(ari, 4),
        "segment_summary": [
            {
                "cluster": int(row["cluster"]),
                "size": int(row["size"]),
                "recency_mean": float(round(row["recency_mean"], 2)),
                "frequency_mean": float(round(row["frequency_mean"], 2)),
                "spend_mean": float(round(row["spend_mean"], 2)),
            }
            for _, row in summary.iterrows()
        ],
    }
    return out


if __name__ == "__main__":
    import sys
    print(json.dumps(run(), indent=2))
