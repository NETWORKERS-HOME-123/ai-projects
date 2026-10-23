import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split


def _pearson_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return df.corr(method="pearson")


def _spearman_matrix(df: pd.DataFrame) -> pd.DataFrame:
    # Spearman is Pearson on ranks
    return df.rank(axis=0).corr(method="pearson")


def _mean_abs_offdiag(mat: pd.DataFrame) -> float:
    a = mat.values.copy()
    n = a.shape[0]
    # exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    return float(np.mean(np.abs(a[mask])))


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the correlation analysis pipeline.

    Returns a JSON-serialisable dict of summary metrics.
    """
    rng = np.random.RandomState(seed)
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # 1. Data checks: missing and simple outlier counts (z-score > 3)
    n_samples, n_features = X.shape
    missing_counts = int(X.isna().sum().sum())
    # z-score per column
    z = (X - X.mean()) / X.std(ddof=0)
    outlier_mask = (np.abs(z) > 3)
    n_outlier_cells = int(outlier_mask.values.sum())
    rows_with_outlier = int((outlier_mask.any(axis=1)).sum())

    # 2. Build experiment: Pearson and Spearman on full data
    pearson_mat = _pearson_matrix(X)
    spearman_mat = _spearman_matrix(X)

    pearson_top_abs = float(np.max(np.abs(pearson_mat.values[np.triu_indices(n_features, k=1)])))
    pearson_mean_abs = _mean_abs_offdiag(pearson_mat)
    spearman_mean_abs = _mean_abs_offdiag(spearman_mat)

    # 3. Held-out and uncertainty: simple split and bootstrap on training data
    X_train, X_holdout = train_test_split(X, test_size=0.2, random_state=seed)

    # bootstrap mean absolute Pearson (on off-diagonal) from resamples of the training set
    n_boot = 200
    boot_means = np.zeros(n_boot)
    n_train = X_train.shape[0]
    for i in range(n_boot):
        idx = rng.randint(0, n_train, size=n_train)
        sample = X_train.iloc[idx]
        pm = _pearson_matrix(sample)
        boot_means[i] = _mean_abs_offdiag(pm)
    bootstrap_pearson_mean_abs = float(np.mean(boot_means))
    bootstrap_pearson_std = float(np.std(boot_means, ddof=1))

    # 4. Outlier sensitivity: recompute Pearson after removing rows with any extreme value
    no_outliers = X[~outlier_mask.any(axis=1)]
    pearson_no_outliers = _pearson_matrix(no_outliers) if len(no_outliers) > 1 else pearson_mat
    pearson_mean_abs_no_outliers = _mean_abs_offdiag(pearson_no_outliers)
    outlier_sensitive_drop = float(pearson_mean_abs - pearson_mean_abs_no_outliers)

    # 5. Quick nonlinear check: compare Pearson vs Spearman top abs
    spearman_top_abs = float(np.max(np.abs(spearman_mat.values[np.triu_indices(n_features, k=1)])))
    nonlinear_flag = bool((spearman_top_abs - pearson_top_abs) > 0.15)

    result = {
        "n_samples": int(n_samples),
        "n_features": int(n_features),
        "missing_cells": missing_counts,
        "n_outlier_cells": n_outlier_cells,
        "rows_with_outlier": rows_with_outlier,
        "pearson_top_abs": pearson_top_abs,
        "pearson_mean_abs": pearson_mean_abs,
        "spearman_mean_abs": spearman_mean_abs,
        "spearman_top_abs": spearman_top_abs,
        "bootstrap_pearson_mean_abs": bootstrap_pearson_mean_abs,
        "bootstrap_pearson_std": bootstrap_pearson_std,
        "pearson_mean_abs_no_outliers": pearson_mean_abs_no_outliers,
        "outlier_sensitive_drop": outlier_sensitive_drop,
        "nonlinear_flag": nonlinear_flag,
        "note": "Pearson = linear; Spearman = rank. Correlation != causation."
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(42), indent=2))
