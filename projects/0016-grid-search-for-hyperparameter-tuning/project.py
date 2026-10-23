import json
from time import perf_counter
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


def run(seed: int = 42) -> Dict:
    # 1) Prepare data: final holdout and training folds
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    # 2) Build pipeline and grid search
    pipe = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(random_state=seed))])

    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [3, None],
        "clf__max_features": ["sqrt", 0.5],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="accuracy",
        cv=cv,
        n_jobs=1,
        return_train_score=False,
    )

    start = perf_counter()
    gs.fit(X_train, y_train)
    total_search_time = perf_counter() - start

    # 3) Evaluate selected estimator on final holdout
    best = gs.best_estimator_
    y_pred = best.predict(X_test)
    test_acc = float(accuracy_score(y_test, y_pred))
    best_cv_score = float(gs.best_score_)

    # 4) Build ranked experiment table from cv_results_
    results = gs.cv_results_
    rows: List[Dict] = []
    for i in range(len(results["params"])):
        row = {
            "params": {k: (v if not isinstance(v, np.generic) else np.asscalar(v)) for k, v in results["params"][i].items()},
            "mean_cv_score": float(results["mean_test_score"][i]),
            "std_cv_score": float(results["std_test_score"][i]),
            "mean_fit_time_sec": float(results["mean_fit_time"][i]),
        }
        rows.append(row)

    # rank by mean_cv_score descending, then by mean_fit_time ascending
    rows_sorted = sorted(rows, key=lambda r: (-r["mean_cv_score"], r["mean_fit_time_sec"]))

    out = {
        "best_params": gs.best_params_,
        "best_cv_score": best_cv_score,
        "test_accuracy": test_acc,
        "total_grid_search_time_sec": float(total_search_time),
        "experiment_table": rows_sorted,
    }

    return out


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
