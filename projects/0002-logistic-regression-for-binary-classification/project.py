import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, confusion_matrix


def run(seed: int = 42) -> Dict:
    """Train a Logistic Regression on synthetic binary data and evaluate thresholds.

    Returns a JSON-serialisable dict with held-out metrics for a chosen threshold (0.5).
    """
    rng = np.random.RandomState(seed)
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=5,
        n_redundant=0,
        n_repeated=0,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        class_sep=1.5,
        flip_y=0.02,
        random_state=rng,
    )

    X = pd.DataFrame(X, columns=[f"feat_{i}" for i in range(X.shape[1])])
    y = pd.Series(y, name="label")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    pipe = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(solver="liblinear", random_state=seed, max_iter=200),
            ),
        ]
    )

    pipe.fit(X_train, y_train)

    probs = pipe.predict_proba(X_test)[:, 1]

    # Evaluate several thresholds to show trade-offs; pick 0.5 as default decision rule
    thresholds = [0.4, 0.5, 0.6]
    threshold_results = {}
    for t in thresholds:
        preds = (probs >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        threshold_results[str(t)] = {
            "precision": float(precision_score(y_test, preds, zero_division=0)),
            "recall": float(recall_score(y_test, preds, zero_division=0)),
            "f1": float(f1_score(y_test, preds, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, preds)),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

    # Main reported metrics use threshold 0.5
    main = threshold_results["0.5"]

    result = {
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "threshold": 0.5,
        "accuracy": main["accuracy"],
        "precision": main["precision"],
        "recall": main["recall"],
        "f1": main["f1"],
        "confusion": {
            "tn": main["tn"],
            "fp": main["fp"],
            "fn": main["fn"],
            "tp": main["tp"],
        },
        "threshold_comparison": threshold_results,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
