import json
from typing import Dict, Any
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score


def run(seed: int = 42) -> Dict[str, Any]:
    """Train a small digit classifier on sklearn's digits dataset.

    Returns a JSON-serialisable dict of metrics and a small suggestions preview.
    """
    # Load data
    data = load_digits()
    X = data.images  # shape (n_samples, 8, 8)
    y = data.target
    vocabulary = [str(i) for i in range(10)]  # documented character vocabulary

    # Create deterministic pseudo-writer groups because the dataset lacks writer ids.
    # We hash a simple statistic (sum of pixels) into 12 groups reproducibly.
    pixel_sums = X.reshape(X.shape[0], -1).sum(axis=1)
    # Use a deterministic bucketing to form groups
    writer_groups = (np.round(pixel_sums).astype(int) % 12).astype(int)

    # Group-aware split: ensure whole pseudo-writers go to train or test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=seed)
    train_idx, test_idx = next(gss.split(X, y, groups=writer_groups))

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]

    # Flatten images and build a small, deterministic pipeline
    n_samples, h, w = X_train.shape
    X_train_flat = X_train.reshape(n_samples, h * w)
    X_test_flat = X_test.reshape(X_test.shape[0], h * w)

    # Drop explicit multi_class keyword for broader sklearn compatibility;
    # solver and max_iter kept for convergence.
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=seed))
    ])

    pipe.fit(X_train_flat, y_train)

    # Predictions and metrics
    y_pred = pipe.predict(X_test_flat)
    y_proba = pipe.predict_proba(X_test_flat)

    accuracy = float(accuracy_score(y_test, y_pred))
    top3 = float(top_k_accuracy_score(y_test, y_proba, k=3))

    # Per-class accuracy mean
    per_class_acc = []
    for cls in np.unique(y_test):
        mask = y_test == cls
        if mask.sum() == 0:
            continue
        per_class_acc.append(accuracy_score(y_test[mask], y_pred[mask]))
    per_class_accuracy_mean = float(np.mean(per_class_acc)) if per_class_acc else 0.0

    # Editable suggestions: for the first 10 test samples provide top-3 candidates and an "Unknown" fallback
    suggestions = []
    unknown_threshold = 0.5
    for i in range(min(10, X_test_flat.shape[0])):
        probs = y_proba[i]
        top3_idx = np.argsort(probs)[-3:][::-1]
        top3_chars = [vocabulary[int(idx)] for idx in top3_idx]
        top_prob = float(probs[top3_idx[0]])
        label = top3_chars[0] if top_prob >= unknown_threshold else "Unknown"
        suggestions.append({"index": int(test_idx[i]), "label": label, "candidates": top3_chars, "top_prob": top_prob})

    unknown_fraction = float((np.max(y_proba, axis=1) < unknown_threshold).mean())

    result = {
        "dataset": "sklearn_digits_8x8",
        "vocabulary": vocabulary,
        "n_train": int(X_train_flat.shape[0]),
        "n_test": int(X_test_flat.shape[0]),
        "accuracy": accuracy,
        "top3_accuracy": top3,
        "per_class_accuracy_mean": per_class_accuracy_mean,
        "unknown_fraction": unknown_fraction,
        "suggestions_preview": suggestions,
    }

    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
