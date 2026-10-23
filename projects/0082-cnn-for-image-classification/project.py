import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def conv2d(images: np.ndarray, kernels: np.ndarray, bias: np.ndarray, stride: int = 1) -> np.ndarray:
    # images: (N, H, W), kernels: (K, kh, kw)
    N, H, W = images.shape
    K, kh, kw = kernels.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    out = np.zeros((N, K, out_h, out_w), dtype=float)
    for i in range(out_h):
        for j in range(out_w):
            patch = images[:, i * stride:i * stride + kh, j * stride:j * stride + kw]
            # patch: (N, kh, kw)
            # compute dot with each kernel
            for k in range(K):
                out[:, k, i, j] = np.tensordot(patch, kernels[k], axes=([1, 2], [0, 1])) + bias[k]
    return out


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def maxpool(feature_maps: np.ndarray, pool: int = 2) -> np.ndarray:
    # feature_maps: (N, K, H, W)
    N, K, H, W = feature_maps.shape
    H2 = H // pool
    W2 = W // pool
    out = np.zeros((N, K, H2, W2), dtype=float)
    for i in range(H2):
        for j in range(W2):
            patch = feature_maps[:, :, i * pool:(i + 1) * pool, j * pool:(j + 1) * pool]
            out[:, :, i, j] = patch.reshape(N, K, -1).max(axis=2)
    return out


def extract_features(images: np.ndarray, seed: int = 42) -> np.ndarray:
    # images: (N, H, W) floats in original scale; we normalise to [0,1]
    rng = np.random.RandomState(seed)
    images = images.astype(float) / images.max()
    # define small set of random kernels for teaching (these are not learned)
    # use 6 kernels of size 3x3
    kernels = rng.randn(6, 3, 3) * 0.5
    bias = np.zeros(6)
    fm = conv2d(images, kernels, bias, stride=1)
    fm = relu(fm)
    fm = maxpool(fm, pool=2)
    # also add a simple orientation-invariant pooled summary
    # global average pooling per feature map
    gap = fm.mean(axis=(2, 3))  # (N, K)
    # flatten local feature maps too
    flat = fm.reshape(fm.shape[0], -1)
    features = np.concatenate([flat, gap], axis=1)
    return features


def run(seed: int = 42) -> Dict[str, Any]:
    # Load data
    data = load_digits()
    X = data.images.copy()  # shape (n_samples, 8, 8)
    y = data.target
    n_classes = len(np.unique(y))

    # Make a deliberate small lighting shift augmentation on a copy for inspection
    # but we keep train/test split clean: we will use the original images.
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Extract features (deterministic given seed)
    feat_train = extract_features(X_train, seed=seed)
    feat_test = extract_features(X_test, seed=seed)

    # Fit a simple classifier
    # some sklearn versions don't accept the multi_class kwarg here, keep defaults for compatibility
    clf = LogisticRegression(max_iter=2000, random_state=seed)
    clf.fit(feat_train, y_train)
    y_pred = clf.predict(feat_test)
    acc = float(accuracy_score(y_test, y_pred))

    # Baseline: most frequent class
    dummy = DummyClassifier(strategy='most_frequent', random_state=seed)
    dummy.fit(feat_train, y_train)
    y_dummy = dummy.predict(feat_test)
    baseline_acc = float(accuracy_score(y_test, y_dummy))

    # Per-class error rates
    per_class_errors = {}
    for cls in range(n_classes):
        mask = (y_test == cls)
        if mask.sum() == 0:
            per_class_errors[str(cls)] = None
        else:
            err = 1.0 - accuracy_score(y_test[mask], y_pred[mask])
            per_class_errors[str(cls)] = float(err)

    # Return metrics (JSON serialisable)
    out = {
        'accuracy': acc,
        'baseline_accuracy': baseline_acc,
        'n_classes': int(n_classes),
        'per_class_error_mean': float(pd.Series(per_class_errors).dropna().astype(float).mean()),
    }
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
