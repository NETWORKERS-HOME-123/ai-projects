import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)
    # 1) Define labels: synthetic measurements with a small defect class
    X, y = make_classification(
        n_samples=800,
        n_features=8,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=1,
        weights=[0.85, 0.15],
        class_sep=1.0,
        flip_y=0.02,
        random_state=rng,
    )

    # train / held-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=rng
    )

    # 2) Scale and fit: pipelines for linear and rbf kernels
    pipes = {
        'linear': Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='linear', probability=False, random_state=seed))]),
        'rbf': Pipeline([('scaler', StandardScaler()), ('svc', SVC(kernel='rbf', C=1.0, gamma='scale', probability=False, random_state=seed))]),
    }

    # 3) Compare boundaries: cross-validation on training folds
    cv_results = {}
    for name, pipe in pipes.items():
        scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1)
        cv_results[name] = {
            'mean_cv_accuracy': float(np.mean(scores)),
            'std_cv_accuracy': float(np.std(scores)),
        }

    # choose best kernel by mean CV accuracy (tie-breaker: linear preferred)
    best_kernel = 'linear' if cv_results['linear']['mean_cv_accuracy'] >= cv_results['rbf']['mean_cv_accuracy'] else 'rbf'
    best_pipe = pipes[best_kernel]
    best_pipe.fit(X_train, y_train)

    # 4) Review failures: evaluate on held-out test set
    y_pred = best_pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    # false negatives: true label == 1 (defect) but predicted 0 (pass)
    fn = int(((y_test == 1) & (y_pred == 0)).sum())
    positives = int((y_test == 1).sum())
    false_negative_rate = float(fn / positives) if positives > 0 else 0.0

    # uncertainty: use distance to decision boundary (absolute decision_function)
    # items with small absolute decision value are uncertain and sent to staff
    try:
        decisions = best_pipe.decision_function(X_test)
        margins = np.abs(decisions)
    except Exception:
        # some SVC settings may not provide decision_function; fall back to None
        margins = np.full(len(X_test), np.inf)
    uncertain_threshold = np.percentile(margins, 15)  # 15% least confident
    uncertain_count = int((margins <= uncertain_threshold).sum())

    results = {
        'kernel_chosen': best_kernel,
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1': float(f1),
        'false_negatives': int(fn),
        'false_negative_rate': float(false_negative_rate),
        'uncertain_count': int(uncertain_count),
        'train_cv_linear_mean': float(cv_results['linear']['mean_cv_accuracy']),
        'train_cv_rbf_mean': float(cv_results['rbf']['mean_cv_accuracy']),
    }
    return results


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
