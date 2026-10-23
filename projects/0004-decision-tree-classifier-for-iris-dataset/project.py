import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the decision tree experiment on Iris.

    Returns a JSON-serialisable dict of metrics and a short explanation string.
    Deterministic when seed is fixed.
    """
    data = load_iris()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Stratified held-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )

    # Unconstrained tree (can overfit)
    tree_full = DecisionTreeClassifier(random_state=seed)
    tree_full.fit(X_train, y_train)

    # Shallow tree for comparison
    tree_shallow = DecisionTreeClassifier(max_depth=2, random_state=seed)
    tree_shallow.fit(X_train, y_train)

    # Predictions
    y_pred = tree_full.predict(X_test)
    y_pred_shallow = tree_shallow.predict(X_test)

    # Metrics
    accuracy = float(accuracy_score(y_test, y_pred))
    accuracy_shallow = float(accuracy_score(y_test, y_pred_shallow))
    precision_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))

    # Per-class precision and recall
    per_class_precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    per_class = {
        f"precision_class_{int(cls)}": float(per_class_precision[i])
        for i, cls in enumerate(np.unique(y_test))
    }
    per_class.update({
        f"recall_class_{int(cls)}": float(per_class_recall[i])
        for i, cls in enumerate(np.unique(y_test))
    })

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred).tolist()

    # Explain a path: choose the first test sample
    sample = X_test.iloc[0:1]
    sample_idx = X_test.index[0]
    rules = export_text(tree_full, feature_names=list(X.columns))
    # Find nodes in the decision path
    node_indicator = tree_full.decision_path(sample)
    node_index = node_indicator.indices
    path_rules = []
    lines = rules.splitlines()
    # export_text prints nodes in order; we select lines for nodes in path if possible
    for nid in node_index:
        # There is no direct mapping from node id to line index in export_text output, keep a simple excerpt
        path_rules.append(f"node_{nid}")
    explanation = (
        "The exported tree rules (excerpt):\n" + rules + "\n"
        "Sample index: "
        + str(int(sample_idx))
        + " took path nodes: "
        + ",".join(str(int(n)) for n in node_index)
        + ".\nSee the rules above for the numeric splits that led to the leaf."
    )

    results: Dict[str, Any] = {
        "accuracy": accuracy,
        "accuracy_shallow": accuracy_shallow,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "confusion_matrix": cm,
        "explanation": explanation,
    }
    results.update(per_class)
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
