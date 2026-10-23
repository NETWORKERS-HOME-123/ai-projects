import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score


def run(seed: int = 42) -> Dict:
    """Generate synthetic customer-like data, fit KMeans for K=2..6,
    pick best K by silhouette, and report cluster diagnostics.
    Deterministic for a given seed.
    """
    rng = np.random.RandomState(seed)
    # Synthetic customer features: purchase_frequency, avg_order_value, recency_days
    X, _ = make_blobs(n_samples=600, centers=4, cluster_std=[1.0, 1.5, 0.8, 1.2],
                      n_features=3, random_state=seed)
    # Shift and scale axes to make them more customer-like and positive
    X = X * np.array([2.0, 1.5, 8.0]) + np.array([3.0, 50.0, 30.0])
    df = pd.DataFrame(X, columns=["purchase_frequency", "avg_order_value", "recency_days"])

    # Feature selection: only aggregated numeric features (already safe)
    features = ["purchase_frequency", "avg_order_value", "recency_days"]
    Xf = df[features].values

    # Scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(Xf)

    best_k = None
    best_sil = -1.0
    best_inertia = None
    best_labels = None
    results = {}

    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        labels = km.fit_predict(Xs)
        # silhouette requires at least 2 clusters and fewer clusters than samples
        sil = silhouette_score(Xs, labels) if k > 1 else -1.0
        if sil > best_sil:
            best_sil = sil
            best_k = k
            best_inertia = float(km.inertia_)
            best_labels = labels.copy()

    # Compute cluster sizes and centroids in original (unscaled) space for interpretability
    km_best = KMeans(n_clusters=best_k, random_state=seed, n_init=10).fit(Xs)
    labels_best = km_best.labels_
    counts = [int((labels_best == i).sum()) for i in range(best_k)]
    centroids_scaled = km_best.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    centroids_list = centroids.tolist()

    # Stability: run KMeans with a couple different seeds and compute ARI vs base run
    seeds = [seed, seed + 1, seed + 2]
    label_sets = []
    for s in seeds:
        km = KMeans(n_clusters=best_k, random_state=s, n_init=10).fit(Xs)
        label_sets.append(km.labels_)
    # ARI between first and others
    ari_1_2 = adjusted_rand_score(label_sets[0], label_sets[1])
    ari_1_3 = adjusted_rand_score(label_sets[0], label_sets[2])
    stability = float((ari_1_2 + ari_1_3) / 2.0)

    # Round floats for stable JSON output
    def R(x, d=6):
        if isinstance(x, float):
            return round(x, d)
        return x

    results = {
        "best_k": int(best_k),
        "silhouette": R(float(best_sil)),
        "inertia": R(float(best_inertia)),
        "cluster_sizes": counts,
        "centroids": [[R(float(v)) for v in row] for row in centroids_list],
        "stability_ari": R(float(stability))
    }
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
