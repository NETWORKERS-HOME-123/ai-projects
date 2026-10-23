import json
import time
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score


def _estimate_memory_kb(*arrays) -> int:
    # Conservative estimate: sum of numpy array nbytes for provided arrays
    total = 0
    for a in arrays:
        if hasattr(a, "nbytes"):
            total += int(a.nbytes)
        else:
            try:
                arr = np.asanyarray(a)
                total += int(arr.nbytes)
            except Exception:
                pass
    # return kilobytes
    return int(total // 1024)


def run(seed: int = 42) -> Dict:
    """Run the size-vs-resource benchmark.

    Returns a JSON-serialisable dict with per-size train times (s), memory estimate (KB) and test accuracy.
    The main summary metric is 'largest_accuracy' (accuracy on the held-out set for the largest dataset).
    """
    rng = np.random.RandomState(seed)

    # sizes chosen to be reasonably quick in test environments but show scaling
    sizes = [2000, 5000, 10000]

    train_times: List[float] = []
    memory_kb: List[int] = []
    accuracies: List[float] = []

    for n_samples in sizes:
        # Create a synthetic classification problem
        X, y = make_classification(
            n_samples=n_samples,
            n_features=15,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=2,
            weights=[0.6, 0.4],
            flip_y=0.01,
            class_sep=1.0,
            random_state=seed + n_samples,
        )

        # Add a simple low-cardinality categorical feature by binning a continuous column
        cat_col = pd.cut(X[:, 0], bins=4, labels=False)
        X_df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
        X_df["cat0"] = cat_col.astype(int)

        # Train / test split with fixed random state
        X_train, X_test, y_train, y_test = train_test_split(
            X_df, y, test_size=0.2, stratify=y, random_state=seed
        )

        # Configure a histogram-based gradient booster (similar pedagogically to LightGBM)
        model = HistGradientBoostingClassifier(
            max_iter=500,
            max_leaf_nodes=16,  # control tree complexity
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=seed,
        )

        # Time and train
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        t1 = time.perf_counter()

        # Memory estimate: sum of numpy bytes for training and test arrays and targets
        mem_kb = _estimate_memory_kb(X_train.values, X_test.values, y_train, y_test)

        # Evaluate
        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))

        train_times.append(float(t1 - t0))
        memory_kb.append(int(mem_kb))
        accuracies.append(acc)

    result = {
        "sizes": sizes,
        "train_times_s": train_times,
        "memory_kb": memory_kb,
        "accuracies": accuracies,
        "largest_accuracy": accuracies[-1],
        "notes": (
            "Used scikit-learn HistGradientBoostingClassifier as LightGBM is not available in the restricted "
            "environment. Memory is a conservative estimate from numpy array nbytes."
        ),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
