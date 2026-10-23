import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score


def run(seed: int = 42) -> Dict:
    # 1. Frame the target
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # 2. Train / test split (fixed, reproducible)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # 3. Fit a RandomForestClassifier (trees do not require scaling)
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)

    # Predict on held-out data
    y_pred = clf.predict(X_test)

    # 4. Baseline: majority-class predictor
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    y_base = baseline.predict(X_test)

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    base_acc = float(accuracy_score(y_test, y_base))
    # Per-class precision/recall for class 0 and 1
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Feature importances (top 5)
    importances = pd.Series(clf.feature_importances_, index=X.columns)
    top5 = importances.sort_values(ascending=False).head(5)
    top_features: List[str] = list(top5.index)
    top_importances: List[float] = [float(x) for x in top5.values]

    result = {
        "accuracy": acc,
        "balanced_accuracy": bal_acc,
        "baseline_accuracy": base_acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "n_features": int(X.shape[1]),
        "top_features": top_features,
        "top_importances": top_importances,
        "note": (
            "Data: sklearn.load_breast_cancer. Model: sklearn.ensemble.RandomForestClassifier. "
            "Do not treat this as maintenance performance; collect labelled maintenance history for real pilots."
        ),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
