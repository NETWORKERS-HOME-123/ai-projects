import json
import time
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


def _to_float(x):
    # ensure JSON-serialisable native floats/ints
    if isinstance(x, (np.floating, float)):
        return float(x)
    if isinstance(x, (np.integer, int)):
        return int(x)
    return x


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the ensemble comparison experiment.

    Returns a JSON-serialisable dict with accuracies, per-class metrics,
    error-overlap counts and a deterministic inference-cost proxy.
    """
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # Define models with fixed random_state where applicable
    tree = DecisionTreeClassifier(random_state=seed)
    rf = RandomForestClassifier(n_estimators=50, random_state=seed)
    lr = LogisticRegression(max_iter=200, random_state=seed)
    knn = KNeighborsClassifier(n_neighbors=5)

    voting = VotingClassifier(
        estimators=[("tree", tree), ("lr", lr), ("knn", knn)], voting="hard"
    )

    # Fit models on the same training split
    tree.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    voting.fit(X_train, y_train)

    # Predictions on held-out set
    preds = {
        "tree": tree.predict(X_test),
        "rf": rf.predict(X_test),
        "voting": voting.predict(X_test),
    }

    # Accuracy per model
    acc = {k: accuracy_score(y_test, v) for k, v in preds.items()}

    # Per-class precision/recall/f1 for each model
    per_class = {}
    for name, yhat in preds.items():
        p, r, f1, support = precision_recall_fscore_support(
            y_test, yhat, zero_division=0
        )
        cls_scores = {}
        for i, cls in enumerate(data.target_names):
            cls_scores[cls] = {
                "precision": _to_float(p[i]),
                "recall": _to_float(r[i]),
                "f1": _to_float(f1[i]),
                "support": int(support[i]),
            }
        per_class[name] = cls_scores

    # Error overlap: which test indices were misclassified by each model
    mis_idx = {
        name: set(np.where(preds[name] != y_test.values)[0]) for name in preds
    }

    # Compute simple overlap counts
    overlap = {}
    model_names = list(preds.keys())
    for i, a in enumerate(model_names):
        for j, b in enumerate(model_names):
            if j <= i:
                continue
            key = f"{a}_and_{b}"
            inter = mis_idx[a].intersection(mis_idx[b])
            union = mis_idx[a].union(mis_idx[b])
            overlap[key] = {
                "both_misclassified": int(len(inter)),
                "either_misclassified": int(len(union)),
                "jaccard": _to_float(len(inter) / len(union) if len(union) > 0 else 0.0),
            }

    # Deterministic inference-cost proxy: use component counts as a cheap, stable proxy
    # (unit_cost_microsec is an arbitrary unit to make numbers readable)
    unit_cost_microsec = 10.0
    inference_cost = {
        "tree_microsec_est": _to_float(1 * unit_cost_microsec),
        "rf_microsec_est": _to_float(rf.n_estimators * unit_cost_microsec),
        "voting_microsec_est": _to_float(len(voting.estimators) * unit_cost_microsec),
    }

    # Assemble report
    report = {
        "accuracy_tree": _to_float(acc["tree"]),
        "accuracy_rf": _to_float(acc["rf"]),
        "accuracy_voting": _to_float(acc["voting"]),
        "per_class": per_class,
        "error_overlap": overlap,
        "inference_cost_proxy": inference_cost,
        "n_test_samples": int(len(y_test)),
        "seed": int(seed),
    }

    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
