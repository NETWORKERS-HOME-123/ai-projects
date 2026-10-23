import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score


def _make_threshold_table(y_true: np.ndarray, scores: np.ndarray, fpr: np.ndarray, tpr: np.ndarray, thresholds: np.ndarray, n_display: int = 10) -> List[Dict]:
    # Select up to n_display thresholds evenly across the ROC thresholds
    if len(thresholds) == 0:
        return []
    idx = np.linspace(0, len(thresholds) - 1, min(n_display, len(thresholds))).astype(int)
    table = []
    for i in idx:
        thr = float(thresholds[i])
        # Predict positive when score >= thr
        yp = (scores >= thr).astype(int)
        fp = int(((yp == 1) & (y_true == 0)).sum())
        fn = int(((yp == 0) & (y_true == 1)).sum())
        table.append({
            "threshold": round(thr, 6),
            "fpr": round(float(fpr[i]), 6),
            "tpr": round(float(tpr[i]), 6),
            "false_alarms": fp,
            "missed_cases": fn,
        })
    return table


def run(seed: int = 42) -> Dict:
    """Train a classifier on synthetic imbalanced data, compute ROC and PR curves.

    Returns a JSON-serialisable dict with the following keys:
      - roc_auc: float (AUC on the held-out test set)
      - average_precision: float (area under PR curve)
      - roc_auc_reversed_labels: float (AUC when test labels are inverted; should be close to 1 - roc_auc)
      - roc_auc_tied_scores: float (AUC when many test scores are tied)
      - positive_label: int (which label is considered positive)
      - threshold_table: list of dict rows with threshold, fpr, tpr, false_alarms, missed_cases
    """
    rng = np.random.RandomState(seed)
    # 1. Prepare data: imbalanced binary classification
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        flip_y=0.01,
        class_sep=1.0,
        random_state=seed,
    )
    # Keep a held-out test set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=seed
    )

    # 2. Build the experiment: simple probabilistic model
    clf = LogisticRegression(solver="liblinear", random_state=seed)
    clf.fit(X_train, y_train)
    # Continuous positive-class scores
    if hasattr(clf, "decision_function"):
        scores = clf.decision_function(X_test)
    else:
        scores = clf.predict_proba(X_test)[:, 1]

    # ROC and AUC
    fpr, tpr, thresholds = roc_curve(y_test, scores, pos_label=1)
    auc = float(roc_auc_score(y_test, scores))

    # Precision-Recall
    precision, recall, pr_thresholds = precision_recall_curve(y_test, scores, pos_label=1)
    ap = float(average_precision_score(y_test, scores))

    # 3. Evaluate reversed labels: invert test labels
    y_test_rev = 1 - y_test
    auc_rev = float(roc_auc_score(y_test_rev, scores))
    # For binary AUC, auc_rev should be approx 1 - auc

    # 3b. Evaluate tied scores: create many identical scores to test ties handling
    scores_tied = np.array(scores)
    # Force middle 30% of scores to a constant value to create ties
    low_q, high_q = np.percentile(scores_tied, [35, 65])
    tie_mask = (scores_tied >= low_q) & (scores_tied <= high_q)
    scores_tied[tie_mask] = float(np.median(scores_tied))
    auc_tied = float(roc_auc_score(y_test, scores_tied))

    # 4. Threshold table: sample thresholds from the ROC thresholds for review
    threshold_table = _make_threshold_table(y_test, scores, fpr, tpr, thresholds, n_display=12)

    result = {
        "roc_auc": round(auc, 6),
        "average_precision": round(ap, 6),
        "roc_auc_reversed_labels": round(auc_rev, 6),
        "roc_auc_tied_scores": round(auc_tied, 6),
        "positive_label": int(1),
        "threshold_table": threshold_table,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
