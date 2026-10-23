import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


def _staged_accuracies(clf: AdaBoostClassifier, X, y) -> List[float]:
    # AdaBoostClassifier provides staged_predict
    return [accuracy_score(y, y_pred) for y_pred in clf.staged_predict(X)]


def run(seed: int = 42) -> Dict:
    """Run the AdaBoost staged experiment.

    Returns a JSON-serialisable dict with metrics.
    """
    rng = np.random.RandomState(seed)

    # 1. Prepare synthetic data
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        class_sep=1.0,
        flip_y=0.0,
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Create noisy training labels by flipping a fraction
    noise_rate = 0.20  # 20% label noise in the noisy experiment
    n_noise = int(noise_rate * y_train.shape[0])
    noisy_y_train = y_train.copy()
    flip_idx = rng.choice(y_train.shape[0], size=n_noise, replace=False)
    noisy_y_train[flip_idx] = 1 - noisy_y_train[flip_idx]

    # 2. Build the experiment: AdaBoost with stumps
    base = DecisionTreeClassifier(max_depth=1, random_state=seed)
    n_estimators = 50

    ada_clean = AdaBoostClassifier(
        estimator=base, n_estimators=n_estimators, random_state=seed
    )
    ada_noisy = AdaBoostClassifier(
        estimator=base, n_estimators=n_estimators, random_state=seed
    )

    ada_clean.fit(X_train, y_train)
    ada_noisy.fit(X_train, noisy_y_train)

    # 3. Evaluate staged predictions on held-out test set
    staged_acc_clean = _staged_accuracies(ada_clean, X_test, y_test)
    staged_acc_noisy = _staged_accuracies(ada_noisy, X_test, y_test)

    best_round_clean = int(np.argmax(staged_acc_clean)) + 1  # rounds are 1-based
    best_accuracy_clean = float(np.max(staged_acc_clean))
    final_accuracy_clean = float(staged_acc_clean[-1])

    best_round_noisy = int(np.argmax(staged_acc_noisy)) + 1
    best_accuracy_noisy = float(np.max(staged_acc_noisy))
    final_accuracy_noisy = float(staged_acc_noisy[-1])

    # 4. Weak learner weights (estimator weights)
    # estimator_weights_ is a numpy array of length n_estimators
    weak_weights = list(map(float, ada_clean.estimator_weights_))

    result = {
        "final_accuracy_clean": final_accuracy_clean,
        "best_accuracy_clean": best_accuracy_clean,
        "best_round_clean": best_round_clean,
        "final_accuracy_noisy": final_accuracy_noisy,
        "best_accuracy_noisy": best_accuracy_noisy,
        "best_round_noisy": best_round_noisy,
        "weak_weights": weak_weights,
        "noise_rate": float(noise_rate),
        "n_estimators": int(n_estimators),
    }

    # Keep deterministic ordering for JSON convenience
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
