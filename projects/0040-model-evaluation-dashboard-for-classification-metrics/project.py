import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


def _to_list(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return [(_to_list(i) if isinstance(i, (np.ndarray, list, tuple)) else i) for i in x]
    return x


def run(seed: int = 42) -> Dict[str, Any]:
    """Train a simple multiclass classifier on iris and return evaluation summaries.

    Returns a JSON-serialisable dict with keys:
    - dataset_version: str
    - accuracy: float
    - macro_f1: float
    - per_class: dict mapping class label -> {precision, recall, f1, support}
    - confusion_matrix: nested lists
    - roc_auc_ovr: float or None
    - exportable_error_summary: list of dicts with id, true, pred, probability_for_pred
    """
    data = load_iris()
    X = data.data
    y = data.target
    class_names = list(map(str, data.target_names))

    # Hold-out split (stratified for class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    clf = LogisticRegression(max_iter=500, random_state=seed)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    # predict_proba gives shape (n_samples, n_classes)
    y_proba = clf.predict_proba(X_test)

    accuracy = float(accuracy_score(y_test, y_pred))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=list(range(len(class_names))), zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(class_names))))

    # ROC AUC OVR: only if we have probability outputs and more than 1 class
    roc_auc_ovr = None
    try:
        y_test_b = label_binarize(y_test, classes=list(range(len(class_names))))
        roc_auc_ovr = float(roc_auc_score(y_test_b, y_proba, average="macro", multi_class="ovr"))
    except Exception:
        roc_auc_ovr = None

    # Build a small exportable error summary: rows where prediction != truth
    ids = [f"row_{i}" for i in range(len(y_test))]
    errors = []
    for i, (rid, true, pred, proba_row) in enumerate(zip(ids, y_test, y_pred, y_proba)):
        if int(true) != int(pred):
            errors.append(
                {
                    "id": rid,
                    "true": int(true),
                    "pred": int(pred),
                    "prob_for_pred": float(proba_row[int(pred)]),
                }
            )

    # Simple verification against a hand-calculated fixture for an edge case:
    # Fixture: three true labels and all predicted as class 0
    # true = [0,1,2], pred = [0,0,0]
    # For this fixture, precision for class 0 = 1/3, recall for class 0 = 1/1, f1 = 1/2
    y_true_fix = np.array([0, 1, 2])
    y_pred_fix = np.array([0, 0, 0])
    p_fix, r_fix, f_fix, s_fix = precision_recall_fscore_support(
        y_true_fix, y_pred_fix, labels=[0, 1, 2], zero_division=0
    )
    # hand calculations
    # class 0: tp=1, fp=2 -> precision=1/3, recall=1/1
    # F1 = 2 * (p*r) / (p + r) = 2*(1/3)*1 / (1/3 + 1) = 0.5
    # class 1: tp=0, fp=0 -> precision=0, recall=0 (zero_division=0 handled)
    # class 2: same as class 1
    # We'll assert that sklearn agrees with these hand calculations numerically
    assert abs(p_fix[0] - (1.0 / 3.0)) < 1e-8
    assert r_fix[0] == 1.0
    assert f_fix[0] == 0.5
    assert p_fix[1] == 0.0 and r_fix[1] == 0.0
    assert p_fix[2] == 0.0 and r_fix[2] == 0.0

    result: Dict[str, Any] = {
        "dataset_version": "sklearn_iris_v1",
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1)),
        "per_class": per_class,
        "confusion_matrix": _to_list(cm),
        "roc_auc_ovr": roc_auc_ovr,
        "exportable_error_summary": errors,
    }

    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
