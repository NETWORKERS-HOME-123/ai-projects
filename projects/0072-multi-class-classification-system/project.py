import json
from typing import Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the multiclass baseline experiment and return JSON-serialisable metrics.

    Returns keys:
      - macro_f1: float
      - per_class_recall_{label}: float for each label
      - confusion_matrix: list[list[int]] labelled in the same order as labels
      - label_mapping: dict mapping label string -> column index for predict_proba
      - example_review: dict with single sample prediction, top alternatives and review_flag
    """
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Preserve the original label names (integers here) and a string form
    labels = list(np.unique(y))
    label_names = [str(int(l)) for l in labels]

    # Stratified holdout to preserve class proportions
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Baseline pipeline: scaling + random forest (handles multiclass)
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=200, random_state=seed))
    ])
    clf.fit(X_train, y_train)

    # Predict probabilities and preserve mapping label -> proba column index
    proba = clf.predict_proba(X_test)
    # scikit-learn's predict_proba columns are ordered by clf.classes_
    proba_labels = [str(int(c)) for c in clf.classes_]
    label_mapping = {lab: int(i) for i, lab in enumerate(proba_labels)}

    # Predictions and metrics
    y_pred = clf.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    per_class_recall = recall_score(y_test, y_pred, average=None)
    per_class_recall_dict = {
        f"per_class_recall_{str(int(cl))}": float(per_class_recall[idx])
        for idx, cl in enumerate(clf.classes_)
    }

    # Confusion matrix with rows=true labels, cols=predicted labels, labelled
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    cm_list = cm.astype(int).tolist()

    # Example reviewable output: take one test row and list top 3 alternatives
    example_idx = 0
    single_proba = proba[example_idx]
    top_indices = np.argsort(single_proba)[::-1][:3]
    top_alternatives = [
        {"label": str(int(clf.classes_[i])), "prob": float(single_proba[i])}
        for i in top_indices
    ]

    # Review path: flag if top probability is below an uncertainty threshold
    uncertainty_threshold = 0.60
    top_prob = float(single_proba[top_indices[0]])
    review_flag = top_prob < uncertainty_threshold

    example_review = {
        "true_label": str(int(y_test.iloc[example_idx])),
        "predicted_label": str(int(y_pred[example_idx])),
        "top_prob": top_prob,
        "top_alternatives": top_alternatives,
        "review_flag": bool(review_flag),
    }

    result: Dict[str, Any] = {
        "macro_f1": macro_f1,
        "confusion_matrix": cm_list,
        "label_order": [str(int(c)) for c in clf.classes_],
        "label_mapping": label_mapping,
        "example_review": example_review,
    }
    result.update(per_class_recall_dict)
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
