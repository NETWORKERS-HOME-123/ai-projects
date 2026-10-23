import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline


def _make_data(seed: int = 42):
    X, y = make_classification(
        n_samples=500,
        n_features=8,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        class_sep=1.2,
        flip_y=0.02,
        random_state=seed,
    )
    # Put features on different units to mimic mixed-units data
    X = X.copy()
    X[:, 0:2] *= 1000.0   # units like metres -> millimetres
    X[:, 2:4] *= 0.01     # units like grams -> kilograms
    X[:, 4:6] *= 1.0      # keep as-is
    X[:, 6:8] *= 100.0    # different magnitude

    # Inject a small number of outliers in specific columns
    rng = np.random.RandomState(seed)
    n_outliers = 6
    idx = rng.choice(np.arange(X.shape[0]), size=n_outliers, replace=False)
    # Make those rows have extreme values in column 1 and 6
    X[idx, 1] += rng.normal(loc=5000, scale=200, size=n_outliers)
    X[idx, 6] += rng.normal(loc=-20000, scale=1000, size=n_outliers)

    feature_names = [f"f{i}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y), list(idx)


def _scaler_params(scaler) -> Dict[str, List[float]]:
    # Convert common scaler params to lists for JSON
    params = {}
    if hasattr(scaler, "mean_"):
        params["mean_"] = list(map(float, scaler.mean_.tolist()))
    if hasattr(scaler, "scale_"):
        params["scale_"] = list(map(float, scaler.scale_.tolist()))
    if hasattr(scaler, "data_min_"):
        params["data_min_"] = list(map(float, scaler.data_min_.tolist()))
    if hasattr(scaler, "data_max_"):
        params["data_max_"] = list(map(float, scaler.data_max_.tolist()))
    if hasattr(scaler, "center_"):
        params["center_"] = list(map(float, scaler.center_.tolist()))
    if hasattr(scaler, "scale_"):
        params["scale_"] = list(map(float, scaler.scale_.tolist()))
    return params


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the experiment. Returns JSON-serialisable dict with metrics and samples.

    The main held-out metric is 'accuracy_standard'.
    """
    X, y, outlier_idx_all = _make_data(seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Map global outlier indices into test rows (for reporting)
    test_index = X_test.index.tolist()
    outlier_test_indices = [int(i) for i in outlier_idx_all if i in test_index]

    scalers = {
        "standard": StandardScaler(),
        "minmax": MinMaxScaler(),
        "robust": RobustScaler()
    }

    results = {}
    saved_transform = {}
    outlier_examples = {}

    for name, scaler in scalers.items():
        # Fit scaler only on training data to avoid leakage
        scaler.fit(X_train.values)
        X_train_s = scaler.transform(X_train.values)
        X_test_s = scaler.transform(X_test.values)

        clf = KNeighborsClassifier(n_neighbors=5)
        clf.fit(X_train_s, y_train.values)
        acc = float(clf.score(X_test_s, y_test.values))
        results[f"accuracy_{name}"] = acc

        # Save scaler parameters derived from training set
        saved_transform[f"scaler_params_{name}"] = _scaler_params(scaler)

        # For each outlier test row, store before/after vectors (as lists)
        examples = []
        for ridx in outlier_test_indices[:6]:
            row = X.loc[ridx].values.astype(float)
            transformed = scaler.transform(row.reshape(1, -1))[0].tolist()
            examples.append({"index": int(ridx), "before": row.tolist(), "after": transformed})
        outlier_examples[name] = examples

    # Choose best scaler by main metric (standard chosen as main for metricFloor)
    best_name = max([("standard", results["accuracy_standard"]),
                     ("minmax", results["accuracy_minmax"]),
                     ("robust", results["accuracy_robust"])], key=lambda x: x[1])[0]

    out: Dict[str, Any] = {}
    out.update(results)
    out["best_scaler"] = best_name
    out["outlier_test_indices"] = outlier_test_indices
    out["outlier_examples"] = outlier_examples
    out["saved_transform"] = saved_transform
    out["n_train"] = int(X_train.shape[0])
    out["n_test"] = int(X_test.shape[0])

    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
