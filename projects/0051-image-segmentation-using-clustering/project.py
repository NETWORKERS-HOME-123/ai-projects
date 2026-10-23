import json
from typing import Dict
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


def _make_synthetic_images(seed: int = 42):
    rng = np.random.RandomState(seed)
    H = W = 64
    imgs = []
    # Image 1: two rectangles and a circle
    img1 = np.zeros((H, W, 3), dtype=np.float32)
    img1[:, :] = np.array([0.9, 0.9, 0.9])  # light grey background
    img1[8:40, 8:28] = np.array([0.8, 0.2, 0.2])  # red rectangle
    img1[20:56, 36:56] = np.array([0.2, 0.2, 0.8])  # blue rectangle
    # circle green
    cy, cx, r = 44, 18, 8
    y, x = np.ogrid[:H, :W]
    mask = (x - cx) ** 2 + (y - cy) ** 2 <= r * r
    img1[mask] = np.array([0.2, 0.8, 0.2])
    imgs.append(img1)
    # Image 2: three coloured blobs via Gaussian spots
    img2 = np.zeros((H, W, 3), dtype=np.float32)
    coords = [(18, 18, [0.9, 0.6, 0.2]), (46, 18, [0.2, 0.9, 0.6]), (32, 46, [0.6, 0.3, 0.9])]
    for cy, cx, col in coords:
        dy = np.arange(H) - cy
        dx = np.arange(W) - cx
        g = np.exp(-(dx[None, :]**2 + dy[:, None]**2) / (2 * (8.0 ** 2)))
        for c in range(3):
            img2[:, :, c] += g * col[c]
    # clip to [0,1]
    img2 = np.clip(img2, 0.0, 1.0)
    imgs.append(img2)
    # stack images into one larger canvas horizontally for more varied pixels
    canvas = np.concatenate(imgs, axis=1)
    return canvas


def run(seed: int = 42) -> Dict:
    """Run a small segmentation-by-clustering experiment.

    Returns a JSON-serialisable dict with keys:
      - silhouette: float (held-out silhouette score on RGB features)
      - inertia: float (KMeans inertia after fitting on train pixels)
      - n_clusters: int (the K used)

    The code uses a held-out pixel split (train/test) with random_state=seed.
    """
    # Build data
    img = _make_synthetic_images(seed=seed)
    H, W, C = img.shape
    pixels = img.reshape(-1, C)

    # Normalise already in [0,1], but ensure float64 for sklearn
    X = pixels.astype(float)

    # Split pixels into train/test for the unsupervised fit/predict protocol
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=seed)

    # Experiment parameters
    n_clusters = 4
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
    km.fit(X_train)
    labels_test = km.predict(X_test)

    # Evaluate clustering on held-out pixels using silhouette score
    # Silhouette requires at least 2 labels present; handle edge cases defensively
    sil = float('nan')
    try:
        if len(np.unique(labels_test)) > 1 and X_test.shape[0] > len(np.unique(labels_test)):
            sil = float(silhouette_score(X_test, labels_test))
        else:
            sil = float('nan')
    except Exception:
        sil = float('nan')

    # Round numeric outputs to avoid tiny floating-point nondeterminism between runs
    # Keep them as floats for the tests
    def _round_or_nan(x, ndigits=8):
        if isinstance(x, float) and np.isnan(x):
            return float('nan')
        return float(round(x, ndigits))

    return {
        "silhouette": _round_or_nan(sil, ndigits=8),
        "inertia": _round_or_nan(float(km.inertia_), ndigits=8),
        "n_clusters": int(n_clusters),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
