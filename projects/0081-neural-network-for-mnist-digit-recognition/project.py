import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


def run(seed: int = 42) -> Dict[str, Any]:
    """Train a small dense network on the scikit-learn digits dataset.

    Returns a JSON-serialisable dict with test accuracy, per-class accuracy,
    split sizes and a minimal model description.
    """
    # Load data (small 8x8 digits as a stand-in for MNIST)
    data = load_digits()
    X = data.data.astype(np.float32)
    y = data.target.astype(int)

    # First split: train+val and test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    # Reserve a validation set from training
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.125, random_state=seed, stratify=y_trainval
    )
    # Sizes
    n_train, n_val, n_test = len(y_train), len(y_val), len(y_test)

    # Standardise pixels (zero mean, unit variance) using training statistics
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    # Small dense network (MLP) as a teaching example of a fully connected NN
    # Use deterministic random_state; keep network compact for quick runs
    clf = MLPClassifier(
        hidden_layer_sizes=(64,),
        activation='relu',
        solver='adam',
        alpha=1e-4,
        batch_size=32,
        learning_rate_init=1e-3,
        max_iter=400,
        random_state=seed,
        verbose=False
    )

    # Fit on training split
    clf.fit(X_train_s, y_train)

    # Evaluate on test split
    y_prob = clf.predict_proba(X_test_s)
    y_pred = np.argmax(y_prob, axis=1)
    overall_acc = float(accuracy_score(y_test, y_pred))

    # Per-class accuracy
    labels = np.unique(y)
    per_class = {}
    for lab in labels:
        mask = (y_test == lab)
        if mask.sum() == 0:
            per_class[str(int(lab))] = None
        else:
            per_class[str(int(lab))] = float(accuracy_score(y_test[mask], y_pred[mask]))

    # Collect a few misclassified examples with predicted confidence
    mis_idx = np.where(y_pred != y_test)[0]
    mis_examples = []
    # Limit the number for compact output
    for i in mis_idx[:6]:
        conf = float(np.max(y_prob[i]))
        mis_examples.append({
            "index": int(i),
            "true": int(y_test[i]),
            "predicted": int(y_pred[i]),
            "confidence": conf
        })

    result = {
        "test_accuracy": overall_acc,
        "per_class_accuracy": per_class,
        "n_train": int(n_train),
        "n_val": int(n_val),
        "n_test": int(n_test),
        "model_hidden_layers": (64,),
        "misclassified_examples": mis_examples,
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
