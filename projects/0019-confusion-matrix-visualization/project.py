import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


def _as_py(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def run(seed: int = 42) -> dict:
    # 1) Prepare data
    data = load_iris()
    X = data.data
    y = data.target
    class_names = list(data.target_names)

    # Keep explicit class ordering (0,1,2 -> set by class_names)
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(y)), test_size=0.3, stratify=y, random_state=seed
    )

    # 2) Build the experiment: simple deterministic model
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Confusion matrices with explicit ordering
    labels = list(range(len(class_names)))
    cm_counts = confusion_matrix(y_test, y_pred, labels=labels)
    # Row-normalised (per-true-class): avoid division by zero
    row_sums = cm_counts.sum(axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_row_norm = np.divide(cm_counts, row_sums, where=row_sums != 0)
        cm_row_norm = np.nan_to_num(cm_row_norm)

    # 3) Evaluate: standard metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro = float(precision_score(y_test, y_pred, average='macro'))
    recall_macro = float(recall_score(y_test, y_pred, average='macro'))

    # Manual check for one class (pick class 0): compute precision and recall by hand
    target_class = 0
    tp = int(cm_counts[target_class, target_class])
    fn = int(cm_counts[target_class, :].sum() - tp)
    fp = int(cm_counts[:, target_class].sum() - tp)
    manual_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    manual_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # 4) Class-error explorer: for each class, counts, rates and a few example mistakes
    explorer: List[Dict[str, Any]] = []
    for cls in labels:
        tp = int(cm_counts[cls, cls])
        total_true = int(cm_counts[cls, :].sum())
        errors = int(total_true - tp)
        recall = float(tp / total_true) if total_true > 0 else 0.0
        # precision for the class (pred==cls)
        predicted_as = int(cm_counts[:, cls].sum())
        precision = float(tp / predicted_as) if predicted_as > 0 else 0.0

        # Collect up to 3 example mistakes: rows in test where true==cls but pred!=cls
        mistakes_idx = [int(i) for i, (t, p) in zip(idx_test, zip(y_test, y_pred)) if t == cls and p != cls]
        # Take up to 3 examples; include a small feature snippet
        examples = []
        for ii in mistakes_idx[:3]:
            feat = X[ii].tolist()
            examples.append({"index": int(ii), "features": feat})

        explorer.append({
            "class_index": int(cls),
            "class_name": class_names[cls],
            "total_true": total_true,
            "true_positive": tp,
            "errors": errors,
            "recall": float(recall),
            "precision": float(precision),
            "example_mistakes": examples,
        })

    # Reconcile totals sanity check
    test_total = int(len(y_test))
    cm_total = int(cm_counts.sum())
    assert test_total == cm_total, "Confusion matrix total must equal number of test rows"

    result = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "confusion_counts": _as_py(cm_counts),
        "confusion_row_normalized": _as_py(cm_row_norm.round(3)),
        "manual_check_class_0": {
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "manual_recall": float(manual_recall),
            "manual_precision": float(manual_precision),
        },
        "class_error_explorer": explorer,
    }

    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
