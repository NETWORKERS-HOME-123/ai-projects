import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _make_source_variant(X, seed: int):
    rng = np.random.RandomState(seed)
    # add light Gaussian noise and a simple linear warp to simulate domain shift
    noise = rng.normal(scale=6.0, size=X.shape)
    warp = X.copy()
    # small shuffled column mixing to mimic a transform (deterministic with seed)
    cols = np.arange(X.shape[1])
    rng.shuffle(cols)
    warp = warp[:, cols]
    return np.clip(warp + noise, 0, 16)


def _pca_features(pca_mean, components, X):
    # Center and project using provided components (components shape: n_components x n_features)
    Xc = X - pca_mean
    return np.dot(Xc, components.T)


def run(seed: int = 42) -> Dict[str, object]:
    """Run the simulated transfer-learning experiment.

    Returns a JSON-serialisable dict with metrics and a short model card.
    """
    rng = np.random.RandomState(seed)
    data = load_digits()
    X = data.data  # shape (n_samples, 64)
    y = data.target

    # Make a source dataset by transforming the digits data (simulated pretraining data)
    X_source = _make_source_variant(X, seed + 1)

    # Split target into train/test (held-out test for evaluation)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Pretrained backbone: PCA fitted on the source dataset
    n_components = 30
    pca_source = PCA(n_components=n_components, svd_solver="randomized", random_state=seed)
    pca_source.fit(X_source)
    mean_s = pca_source.mean_
    comp_s = pca_source.components_

    # Frozen backbone: use source PCA unchanged
    Xf_train = _pca_features(mean_s, comp_s, X_train)
    Xf_test = _pca_features(mean_s, comp_s, X_test)

    # use a solver that supports multiclass; avoid arguments not present in all sklearn versions
    clf_frozen = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
    clf_frozen.fit(Xf_train, y_train)
    acc_frozen = float(accuracy_score(y_test, clf_frozen.predict(Xf_test)))

    # Fine-tune simulation: compute PCA on target train and blend components (small alpha)
    pca_target = PCA(n_components=n_components, svd_solver="randomized", random_state=seed + 1)
    pca_target.fit(X_train)
    mean_t = pca_target.mean_
    comp_t = pca_target.components_

    alpha = 0.20  # small adaptation step (simulated low learning rate)
    mean_ft = (1 - alpha) * mean_s + alpha * mean_t
    comp_ft = (1 - alpha) * comp_s + alpha * comp_t

    Xft_train = _pca_features(mean_ft, comp_ft, X_train)
    Xft_test = _pca_features(mean_ft, comp_ft, X_test)

    clf_ft = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
    clf_ft.fit(Xft_train, y_train)
    acc_finetune = float(accuracy_score(y_test, clf_ft.predict(Xft_test)))

    # From-scratch baseline: PCA trained only on target train
    pca_scratch = PCA(n_components=n_components, svd_solver="randomized", random_state=seed + 2)
    pca_scratch.fit(X_train)
    Xs_train = pca_scratch.transform(X_train)
    Xs_test = pca_scratch.transform(X_test)

    clf_scratch = LogisticRegression(max_iter=300, solver="lbfgs", random_state=seed)
    clf_scratch.fit(Xs_train, y_train)
    acc_scratch = float(accuracy_score(y_test, clf_scratch.predict(Xs_test)))

    # Tiny proxy for model cost: number of parameters in the linear head (n_classes * n_components)
    n_classes = len(np.unique(y))
    params_head = int(n_classes * n_components)

    # Produce a short model card describing provenance and permitted uses
    model_card = (
        "Model card:\n"
        "Simulated pretrained PCA fitted on a transformed copy of sklearn.datasets.load_digits.\n"
        "Preprocessing: input pixels are 0-16 greyscale; we centre and project with PCA.\n"
        "Checkpoint provenance: simulated (no real external checkpoint).\n"
        "Permitted uses: research and teaching on small-image transfer simulations.\n"
        "Not for deployment on real image tasks without replacing PCA with a real pretrained CNN and"
        " checking licence and data overlap."
    )

    # Round metrics to sensible decimals for determinism in tests
    out = {
        "accuracy_frozen": round(acc_frozen, 6),
        "accuracy_finetune": round(acc_finetune, 6),
        "accuracy_scratch": round(acc_scratch, 6),
        "n_components": int(n_components),
        "params_head": params_head,
        "model_card": model_card,
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
