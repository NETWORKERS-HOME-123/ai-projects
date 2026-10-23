import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, mean_squared_error


def run(seed: int = 42) -> Dict:
    """Run the PCA experiment on the wine dataset.

    Returns a JSON-serialisable dict with numeric metrics and small summaries.
    """
    # Load data
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    n_features = X.shape[1]

    # Split (stratify to keep class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Fit scaler only on training rows
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # Baseline classifier (no PCA)
    # Use a solver that supports multiclass (wine has 3 classes)
    # Some sklearn versions don't accept the multi_class keyword in the constructor,
    # so omit it and rely on the solver's default behavior.
    clf_baseline = LogisticRegression(random_state=seed, max_iter=2000, solver='lbfgs')
    clf_baseline.fit(X_train_s, y_train)
    y_pred_base = clf_baseline.predict(X_test_s)
    baseline_acc = float(accuracy_score(y_test, y_pred_base))

    # Choose several component counts to explore
    max_try = min(n_features, 10)
    components_tried: List[int] = list(range(1, max_try + 1))

    explained_cumsum_by_k = {}
    acc_by_k = {}
    recon_mse_by_k = {}

    for k in components_tried:
        # PCA does not need a random_state for deterministic SVD in this context
        pca = PCA(n_components=k)
        X_train_p = pca.fit_transform(X_train_s)
        X_test_p = pca.transform(X_test_s)

        # Cumulative explained variance for these k components
        cumsum = float(np.sum(pca.explained_variance_ratio_))
        explained_cumsum_by_k[str(k)] = cumsum

        # Reconstruction (from reduced space back to original scaled space)
        X_test_rec_s = pca.inverse_transform(X_test_p)
        mse = float(mean_squared_error(X_test_s, X_test_rec_s))
        recon_mse_by_k[str(k)] = mse

        # Downstream classifier trained on the PCA representation
        clf = LogisticRegression(random_state=seed, max_iter=2000, solver='lbfgs')
        clf.fit(X_train_p, y_train)
        y_pred = clf.predict(X_test_p)
        acc = float(accuracy_score(y_test, y_pred))
        acc_by_k[str(k)] = acc

    # Pick a representative k for reporting (5 or min(5, n_features))
    k_rep = min(5, n_features)
    key_k = str(k_rep)

    out = {
        "n_features": int(n_features),
        "baseline_accuracy": baseline_acc,
        "accuracy_pca_5": acc_by_k[key_k],
        "reconstruction_mse_5": recon_mse_by_k[key_k],
        "explained_variance_cumsum_5": explained_cumsum_by_k[key_k],
        "components_tried": components_tried,
        # include full maps as compact JSON-friendly dicts
        "accuracy_by_k": acc_by_k,
        "reconstruction_mse_by_k": recon_mse_by_k,
        "explained_variance_cumsum_by_k": explained_cumsum_by_k,
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
