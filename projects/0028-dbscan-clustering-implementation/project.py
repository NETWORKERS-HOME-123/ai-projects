import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)

    # 1) Prepare data: create blobs with different stds and some isolated points
    centers = [(-5, 0), (0, 0), (5, 0), (2.5, 4)]
    cluster_std = [0.4, 1.0, 0.6, 0.25]  # differing densities
    X, y = make_blobs(n_samples=[200, 220, 160, 80], centers=centers,
                      cluster_std=cluster_std, random_state=seed)

    # Add some isolated points (likely noise)
    isolated = rng.uniform(low=-10, high=10, size=(20, 2))
    X = np.vstack([X, isolated])

    # Inspect scale (not printed here, but useful for choosing eps range)
    pairwise_dists = np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2))
    median_dist = float(np.median(pairwise_dists))

    # 2) Build the experiment: sweep small ranges of eps and min_samples
    eps_values = [0.3, 0.5, 0.8, 1.1]
    min_samples_values = [3, 5, 8]

    explorer: List[Dict] = []

    best_sil = float("nan")
    best_record = None

    for eps in eps_values:
        for min_s in min_samples_values:
            db = DBSCAN(eps=eps, min_samples=min_s)
            labels = db.fit_predict(X)

            # cluster sizes (exclude noise label = -1 for reporting clusters)
            labels_series = pd.Series(labels)
            cluster_counts = labels_series.value_counts().to_dict()
            n_noise = int(cluster_counts.get(-1, 0))
            n_points = X.shape[0]
            noise_fraction = n_noise / n_points

            # number of non-noise clusters
            unique_clusters = [lab for lab in np.unique(labels) if lab != -1]
            n_clusters = len(unique_clusters)

            # silhouette: only valid when at least 2 clusters
            sil = None
            if n_clusters >= 2:
                try:
                    sil = float(silhouette_score(X, labels))
                except Exception:
                    sil = None

            record = {
                "eps": float(eps),
                "min_samples": int(min_s),
                "n_clusters": int(n_clusters),
                "noise_fraction": float(noise_fraction),
                "cluster_counts": {int(k): int(v) for k, v in cluster_counts.items()},
                "silhouette": (None if sil is None else float(round(sil, 6)))
            }

            explorer.append(record)

            if sil is not None:
                if best_record is None or sil > best_sil:
                    best_sil = sil
                    best_record = record

    # If no silhouette was computable (unlikely here), pick smallest noise fraction
    if best_record is None:
        best_record = min(explorer, key=lambda r: r["noise_fraction"])
        best_sil = float("nan")

    result = {
        "median_pairwise_distance": float(round(median_dist, 6)),
        "best_silhouette": (None if np.isnan(best_sil) else float(round(best_sil, 6))),
        "best_eps": best_record["eps"],
        "best_min_samples": best_record["min_samples"],
        "best_n_clusters": best_record["n_clusters"],
        "best_noise_fraction": best_record["noise_fraction"],
        "explorer": explorer,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
