import json
from typing import Dict, List, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score


def _viewer_payload(images: np.ndarray, idx: int) -> Dict[str, Any]:
    """Create a small JSON-serialisable viewer payload for a single digit image.

    We include the 8x8 pixel grid as a list of ints and supply a simple
    'dataset boundary' marker: a one-pixel thick border indicated in a
    separate mask array (1 for boundary, 0 elsewhere)."""
    img = images[idx]
    if img.ndim != 1:
        flat = img.ravel()
    else:
        flat = img
    pixels = [int(float(x)) for x in flat.tolist()]
    # Build boundary mask for an 8x8 image
    w = 8
    h = 8
    mask = []
    for r in range(h):
        for c in range(w):
            if r == 0 or r == h - 1 or c == 0 or c == w - 1:
                mask.append(1)
            else:
                mask.append(0)
    return {
        "image_shape": [h, w],
        "image_pixels": pixels,
        "boundary_mask": mask,
        "note": "Pixels are 0-16 grayscale. boundary_mask marks the visible dataset boundary."
    }


def run(seed: int = 42) -> Dict[str, Any]:
    # 1. Load data and create a stratified holdout
    digits = load_digits()
    X = digits.data  # shape (n_samples, 64)
    y = digits.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # 2. Train a Random Forest on flattened pixels
    clf = RandomForestClassifier(n_estimators=300, random_state=seed)
    clf.fit(X_train, y_train)

    # 3. Evaluate
    y_pred = clf.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    per_class_recall = recall_score(y_test, y_pred, average=None)
    mean_recall = float(per_class_recall.mean())

    # confusion matrix and top confused pairs
    cm = confusion_matrix(y_test, y_pred)
    # find off-diagonal largest confusions
    n_classes = cm.shape[0]
    confused = []
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j:
                confused.append(((i, j), int(cm[i, j])))
    confused.sort(key=lambda x: x[1], reverse=True)
    top_confused = [ {"true": int(pair[0]), "pred": int(pair[1]), "count": cnt} for (pair, cnt) in confused[:5] ]

    # For inspection, collect a few example indices of mistakes (true != pred)
    mistakes = []
    for idx, (t, p) in enumerate(zip(y_test, y_pred)):
        if t != p:
            mistakes.append({"index_in_test": int(idx), "true": int(t), "pred": int(p)})
            if len(mistakes) >= 6:
                break

    # 4. Build a digits-only prediction viewer payload from the first test image
    viewer = _viewer_payload(X_test, 0)

    # Prepare per-class recall as a dict
    per_class_recall_dict = {str(i): float(r) for i, r in enumerate(per_class_recall)}

    out = {
        "accuracy": acc,
        "mean_recall": mean_recall,
        "per_class_recall": per_class_recall_dict,
        "top_confused_pairs": top_confused,
        "example_mistakes": mistakes,
        "viewer_sample": viewer,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0])
    }
    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
