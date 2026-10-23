import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances


def _run_tsne(X: np.ndarray, perplexity: float, seed: int) -> np.ndarray:
    ts = TSNE(n_components=2, perplexity=perplexity, random_state=seed, init="pca")
    emb = ts.fit_transform(X)
    return emb


def _distance_correlation(X: np.ndarray, Y: np.ndarray) -> float:
    # Compute Pearson correlation between flattened pairwise distances.
    dx = pairwise_distances(X)
    dy = pairwise_distances(Y)
    v1 = dx[np.triu_indices_from(dx, k=1)]
    v2 = dy[np.triu_indices_from(dy, k=1)]
    if v1.std() == 0 or v2.std() == 0:
        return 0.0
    return float(np.corrcoef(v1, v2)[0, 1])


def run(seed: int = 42) -> Dict:
    """Run the t-SNE experiments and return JSON-serialisable metrics.

    Returns a dict with per-run trustworthiness and distance-correlation,
    plus summary statistics. All randomness is driven by the seed argument.
    """
    rng = np.random.RandomState(seed)

    data = load_iris()
    X = data.data.copy()
    y = data.target.copy()

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    perplexities = [5.0, 30.0, 50.0]
    seeds = [seed, seed + 1, seed + 2]

    runs: List[Dict] = []

    # For stability by perplexity: keep embeddings per perplexity across seeds
    embeddings_by_perp = {p: [] for p in perplexities}

    for p in perplexities:
        for s in seeds:
            emb = _run_tsne(Xs, perplexity=p, seed=int(s))
            tw = float(trustworthiness(Xs, emb, n_neighbors=10))
            dc = _distance_correlation(Xs, emb)
            runs.append({"perplexity": float(p), "seed": int(s), "trustworthiness": tw, "distance_correlation": dc})
            embeddings_by_perp[p].append(emb)

    # Compute stability per perplexity as mean pairwise distance-correlation between seed embeddings
    stability = {}
    for p, embs in embeddings_by_perp.items():
        vals = []
        for i in range(len(embs)):
            for j in range(i + 1, len(embs)):
                vals.append(_distance_correlation(embs[i], embs[j]))
        stability[float(p)] = float(np.mean(vals)) if vals else 0.0

    trust_vals = [r["trustworthiness"] for r in runs]
    avg_trust = float(np.mean(trust_vals))

    result = {
        "dataset": "iris",
        "n_samples": int(Xs.shape[0]),
        "n_features": int(Xs.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn_version,
        "perplexities": [float(p) for p in perplexities],
        "seeds": [int(s) for s in seeds],
        "runs": runs,
        "stability": stability,
        "avg_trustworthiness": avg_trust,
        "warning": "t-SNE emphasises local neighbourhoods; global distances and areas are not reliably preserved. Use trustworthiness and stability metrics when interpreting layouts."
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
