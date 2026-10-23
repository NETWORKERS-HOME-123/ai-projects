import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def _make_variation(img: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    noise = rng.normal(scale=2.0, size=img.shape)
    scale = 0.95 + 0.1 * rng.rand()
    out = img * scale + noise
    return np.clip(out, 0.0, 16.0)


def run(seed: int = 42) -> Dict:
    """Build a small image retrieval index over sklearn digits and evaluate.

    Returns a JSON-serialisable dict with metrics and sizes.
    """
    rng = np.random.RandomState(seed)
    digits = load_digits()
    images = digits.images  # shape (n_samples, 8,8)
    n_orig = images.shape[0]

    # Build a catalogue by making 3 variants per original image (including the original)
    catalogue_rows: List[Dict] = []
    for orig_idx in range(n_orig):
        base = images[orig_idx]
        for dup_id in range(3):
            if dup_id == 0:
                img = base.copy()
            else:
                img = _make_variation(base, seed + orig_idx * 3 + dup_id)
            row = {
                "catalogue_id": len(catalogue_rows),
                "original_index": int(orig_idx),
                "dup_id": int(dup_id),
                "image": img.ravel().astype(float),
            }
            catalogue_rows.append(row)

    catalogue = pd.DataFrame(catalogue_rows)
    X_catalogue = np.stack(catalogue["image"].values)

    # Build embeddings using PCA
    pca = PCA(n_components=32, random_state=seed)
    embeddings = pca.fit_transform(X_catalogue)

    # Build nearest-neighbour index
    nn = NearestNeighbors(n_neighbors=10, algorithm="brute", metric="euclidean")
    nn.fit(embeddings)

    # Prepare a small labelled query set: sample 20 original indices and perturb them
    orig_indices = rng.choice(n_orig, size=20, replace=False)
    queries = []
    for i, orig in enumerate(orig_indices):
        q_img = _make_variation(images[orig], seed + 1000 + i)
        queries.append({"query_id": i, "original_index": int(orig), "image": q_img.ravel().astype(float)})
    queries_df = pd.DataFrame(queries)
    X_queries = np.stack(queries_df["image"].values)
    q_emb = pca.transform(X_queries)

    # Retrieve and evaluate recall@k and MRR
    neigh_dist, neigh_idx = nn.kneighbors(q_emb, n_neighbors=10, return_distance=True)
    recalls_at_1 = []
    recalls_at_5 = []
    rr_list = []
    ranked_provenance = []

    for qi, row in enumerate(queries_df.itertuples()):
        true_orig = int(row.original_index)
        neighbours = neigh_idx[qi]
        neigh_meta = catalogue.iloc[neighbours][["catalogue_id", "original_index", "dup_id"]].to_dict(orient="records")
        ranked_provenance.append({"query_id": int(row.query_id), "true_original": true_orig, "neighbours": neigh_meta})

        # compute recall@1 and recall@5
        found_positions = [pos for pos, n in enumerate(neighbours) if int(catalogue.iloc[n]["original_index"]) == true_orig]
        recall1 = 1.0 if (0 in found_positions) else 0.0
        recall5 = 1.0 if any(p < 5 for p in found_positions) else 0.0
        recalls_at_1.append(recall1)
        recalls_at_5.append(recall5)
        rr = 0.0
        if found_positions:
            rr = 1.0 / (1 + min(found_positions))
        rr_list.append(rr)

    recall_at_1 = float(np.mean(recalls_at_1))
    recall_at_5 = float(np.mean(recalls_at_5))
    mrr = float(np.mean(rr_list))

    result = {
        "recall_at_1": round(recall_at_1, 4),
        "recall_at_5": round(recall_at_5, 4),
        "mrr": round(mrr, 4),
        "catalogue_size": int(len(catalogue)),
        "query_size": int(len(queries_df)),
        "method": "PCA(32)+NearestNeighbours(brute,euclidean)",
        "seed": int(seed),
        "ranked_provenance_sample": ranked_provenance[:3],
    }
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(run(42), indent=2))
