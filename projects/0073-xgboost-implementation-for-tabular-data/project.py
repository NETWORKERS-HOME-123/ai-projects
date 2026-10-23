import json
import time
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss, accuracy_score


def run(seed: int = 42) -> Dict:
    """Run the experiment and return a JSON-serialisable dict of results.

    Notes:
    - Uses synthetic data created with sklearn.make_classification.
    - Introduces missing values (5% of entries) to emulate real data.
    - Uses HistGradientBoostingClassifier as a stand-in for XGBoost (xgboost package not available).
    """
    np.random.seed(seed)

    # 1) Build synthetic dataset
    X, y = make_classification(
        n_samples=1000,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        flip_y=0.02,
        class_sep=1.0,
        random_state=seed,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X = pd.DataFrame(X, columns=feature_names)

    # introduce 5% missing values at random
    missing_mask = np.random.rand(*X.shape) < 0.05
    X = X.mask(missing_mask)

    # 2) Train/test split (hold-out test set untouched)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # preprocessing: simple mean imputation (fitted on training data only)
    imputer = SimpleImputer(strategy="mean")
    X_train_imp = imputer.fit_transform(X_train)
    X_test_imp = imputer.transform(X_test)

    results = {
        "n_samples": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "feature_names": feature_names,
    }

    # 3) Train HistGradientBoostingClassifier (stand-in for XGBoost)
    # Use early_stopping with a validation fraction
    gbc = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.1,
        max_leaf_nodes=31,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=seed,
    )

    t0 = time.perf_counter()
    gbc.fit(X_train_imp, y_train)
    t1 = time.perf_counter()
    gbc_time = t1 - t0

    # 4) Train RandomForest baseline with similar budget constraints
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, random_state=seed)
    t0 = time.perf_counter()
    rf.fit(X_train_imp, y_train)
    t1 = time.perf_counter()
    rf_time = t1 - t0

    # 5) Evaluate on held-out test set
    y_prob_gbc = gbc.predict_proba(X_test_imp)[:, 1]
    y_prob_rf = rf.predict_proba(X_test_imp)[:, 1]
    y_pred_gbc = (y_prob_gbc >= 0.5).astype(int)
    y_pred_rf = (y_prob_rf >= 0.5).astype(int)

    auc_gbc = float(roc_auc_score(y_test, y_prob_gbc))
    auc_rf = float(roc_auc_score(y_test, y_prob_rf))
    brier_gbc = float(brier_score_loss(y_test, y_prob_gbc))
    brier_rf = float(brier_score_loss(y_test, y_prob_rf))
    acc_gbc = float(accuracy_score(y_test, y_pred_gbc))
    acc_rf = float(accuracy_score(y_test, y_pred_rf))

    # 6) Prepare a short reproducible report
    pkg_versions = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }

    model_report = {
        "boosted_model": "HistGradientBoostingClassifier (scikit-learn, used as stand-in for XGBoost)",
        "boosted_params": {
            "max_iter": gbc.max_iter,
            "learning_rate": gbc.learning_rate,
            "max_leaf_nodes": gbc.max_leaf_nodes,
            "early_stopping": gbc.early_stopping,
            "n_iter_no_change": gbc.n_iter_no_change,
        },
        "baseline_model": "RandomForestClassifier",
        "baseline_params": {"n_estimators": rf.n_estimators},
    }

    results.update(
        {
            "auc_test": auc_gbc,
            "brier_test": brier_gbc,
            "accuracy_test": acc_gbc,
            "auc_baseline": auc_rf,
            "brier_baseline": brier_rf,
            "accuracy_baseline": acc_rf,
            "time_seconds_boosted": float(round(gbc_time, 4)),
            "time_seconds_baseline": float(round(rf_time, 4)),
            "packages": pkg_versions,
            "report": model_report,
        }
    )

    return results


if __name__ == "__main__":
    import sys

    out = run()
    print(json.dumps(out, indent=2))
