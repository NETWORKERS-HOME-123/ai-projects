import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.utils import resample
from sklearn.decomposition import PCA


def _build_linkage_from_model(model: AgglomerativeClustering) -> List[List[float]]:
    # Build a scipy-style linkage matrix from children_ and distances_
    # children_ shape (n_nodes, 2), distances_ shape (n_nodes,)
    children = model.children_
    n_samples = model.n_leaves_
    if not hasattr(model, "distances_"):
        # distances_ only available when distance_threshold is used
        return []
    distances = model.distances_
    counts = np.zeros(children.shape[0], dtype=int)
    for i, (left, right) in enumerate(children):
        count_left = 1 if left < n_samples else counts[left - n_samples]
        count_right = 1 if right < n_samples else counts[right - n_samples]
        counts[i] = count_left + count_right
    # linkage matrix rows: [idx1, idx2, dist, sample_count]
    linkage = []
    for i, (left, right) in enumerate(children):
        linkage.append([int(left), int(right), float(distances[i]), int(counts[i])])
    return linkage


def _representatives(X: np.ndarray, labels: np.ndarray, n_repr: int = 2) -> Dict[int, List[int]]:
    # For each cluster, pick n_repr points closest to the cluster centroid
    reps = {}
    for lab in np.unique(labels):
        members = np.where(labels == lab)[0]
        if members.size == 0:
            reps[int(lab)] = []
            continue
        centroid = X[members].mean(axis=0)
        dists = np.linalg.norm(X[members] - centroid, axis=1)
        order = np.argsort(dists)[:n_repr]
        reps[int(lab)] = [int(members[i]) for i in order]
    return reps


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the hierarchical clustering experiment and return JSON-serialisable metrics.

    Returns a dict containing silhouette scores for average, complete and Ward linkage,
    a dendrogram-style linkage matrix for Ward, stability (mean ARI over resamples) for Ward,
    and representative indices for a chosen flat cut (n_clusters=3).
    """
    rng = np.random.RandomState(seed)
    data = load_iris()
    X = data.data.astype(float)
    # Step 1: scale features
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # Optional dimensionality reduction to give clearer clusters for teaching
    # Using PCA to 2 components often improves separability for iris and keeps determinism
    pca = PCA(n_components=2)
    Xp = pca.fit_transform(Xs)

    results: Dict[str, Any] = {}

    # We'll evaluate clusterings using a fixed n_clusters = 3 (educational choice)
    n_clusters = 3

    linkages = ["average", "complete", "ward"]
    silhouettes = {}

    for link in linkages:
        # For ward linkage sklearn requires affinity='euclidean' and uses variance minimisation
        # Use AgglomerativeClustering with n_clusters to get flat clustering
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=link)
        # use PCA-transformed data for clustering to improve separability on iris
        labels = model.fit_predict(Xp)
        # Silhouette requires at least 2 clusters and no singletons; silence errors by try/except
        try:
            sil = float(silhouette_score(Xp, labels))
        except Exception:
            sil = float('nan')
        silhouettes[f"silhouette_{link}"] = sil

    results.update(silhouettes)

    # Build a full tree for Ward to allow dendrogram reconstruction
    ward_full = AgglomerativeClustering(distance_threshold=0.0, n_clusters=None, linkage="ward")
    ward_full.fit(Xp)
    linkage_matrix = _build_linkage_from_model(ward_full)
    # Provide a small explorer for a chosen flat cut (n_clusters)
    ward_cut = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
    labels_ward = ward_cut.fit_predict(Xp)
    reps = _representatives(Xp, labels_ward, n_repr=2)

    results["dendrogram_linkage_ward"] = linkage_matrix  # list of merges
    results["representative_indices_ward"] = reps

    # Step: cluster stability under resampling (bootstrap rows) for Ward
    n_boot = 20
    aris = []
    for i in range(n_boot):
        sample_idx = resample(np.arange(Xp.shape[0]), replace=True, random_state=rng.randint(0, 10_000))
        Xb = Xp[sample_idx]
        # Fit clusters on bootstrap sample and map back to original indices via nearest-centroid assignment
        model_b = AgglomerativeClustering(n_clusters=n_clusters, linkage="ward")
        labels_b = model_b.fit_predict(Xb)
        # Build centroids from bootstrap clusters
        centroids = np.vstack([Xb[labels_b == k].mean(axis=0) if np.any(labels_b == k) else np.full(Xp.shape[1], np.nan)
                               for k in range(n_clusters)])
        # Assign original points to nearest centroid (skip nan centroids)
        # If centroid is nan (empty cluster in bootstrap), assign -1
        assigned = np.full(Xp.shape[0], -1)
        valid_centroids = ~np.isnan(centroids).any(axis=1)
        if not valid_centroids.any():
            aris.append(0.0)
            continue
        valid_idxs = np.where(valid_centroids)[0]
        dists = np.linalg.norm(Xp[:, None, :] - centroids[valid_idxs][None, :, :], axis=2)
        assigned = valid_idxs[np.argmin(dists, axis=1)]
        # Compute adjusted rand score between ward labels and assigned labels
        try:
            ari = adjusted_rand_score(labels_ward, assigned)
        except Exception:
            ari = 0.0
        aris.append(float(ari))

    mean_ari = float(np.mean(aris))
    results["stability_ward_mean_ari"] = mean_ari

    # Ensure deterministic ordering for JSON by converting numpy types to native
    # The silhouettes are native floats already; ensure linkage entries are lists
    results["n_samples"] = int(Xp.shape[0])
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
