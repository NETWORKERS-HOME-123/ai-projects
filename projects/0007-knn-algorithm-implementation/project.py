import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


def hand_knn_predict(X_train: np.ndarray, y_train: np.ndarray, X_query: np.ndarray, k: int) -> np.ndarray:
    # Simple, deterministic Euclidean neighbour search
    # X_train: (n_train, n_features), X_query: (n_query, n_features)
    dists = np.sqrt(((X_query[:, None, :] - X_train[None, :, :]) ** 2).sum(axis=2))
    nearest_idx = np.argsort(dists, axis=1)[:, :k]
    # majority vote
    preds = []
    for rows in nearest_idx:
        vals, counts = np.unique(y_train[rows], return_counts=True)
        preds.append(vals[np.argmax(counts)])
    return np.array(preds)


def run(seed: int = 42) -> Dict:
    # 1. Create a small synthetic item-descriptor dataset
    X, y = make_blobs(n_samples=300, centers=3, n_features=6, cluster_std=[1.2, 1.6, 1.0], random_state=seed)

    # 2. Split into train / test and scale using training data only
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 3. Tune k using cross-validation on the training set only
    ks = [1, 3, 5, 7, 9]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    best_k = None
    best_score = -1.0
    for k in ks:
        clf = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring="accuracy", n_jobs=1)
        mean = float(scores.mean())
        if mean > best_score:
            best_score = mean
            best_k = k

    # Fit final KNN with the chosen k
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    test_accuracy = float(accuracy_score(y_test, y_pred))

    # 4. Hand-built neighbour search (same scaling) and inspect misleading nearest neighbours
    hand_preds = hand_knn_predict(X_train_scaled, y_train, X_test_scaled, k=best_k)
    hand_acc = float(accuracy_score(y_test, hand_preds))
    # A simple notion of misleading: nearest neighbour (k=1) label disagrees with true label
    nn1 = hand_knn_predict(X_train_scaled, y_train, X_test_scaled, k=1)
    misleading_count = int((nn1 != y_test).sum())

    return {
        "accuracy": test_accuracy,
        "best_k": int(best_k),
        "cv_best_score": float(best_score),
        "hand_knn_accuracy": hand_acc,
        "misleading_nearest_count": misleading_count,
        "n_test": int(X_test.shape[0])
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
