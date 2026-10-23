import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, learning_curve


def run(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    data = load_wine()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    # Determine number of CV folds so each class has at least one sample per fold
    class_counts = y.value_counts().values
    max_folds = int(class_counts.min())
    cv_folds = min(5, max(2, max_folds))

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    # Choose train sizes as fractions but ensure absolute sizes are sensible
    n_samples = X.shape[0]
    # start at 20% or at least (n_classes * 2) samples, whichever is larger
    min_fraction = max(0.2, (len(np.unique(y)) * 2) / n_samples)
    train_sizes = np.linspace(min_fraction, 1.0, 5)

    # Define two pipelines: simple and complex
    # Use a multiclass-capable solver for logistic regression
    simple = Pipeline([("scale", StandardScaler()),
                       ("clf", LogisticRegression(max_iter=500, solver="lbfgs", random_state=seed))])
    complex_p = Pipeline([("scale", StandardScaler()),
                          ("clf", RandomForestClassifier(n_estimators=200, random_state=seed))])

    results = {}

    for name, estimator in [("logistic", simple), ("random_forest", complex_p)]:
        train_sizes_abs, train_scores, valid_scores = learning_curve(
            estimator,
            X,
            y,
            train_sizes=train_sizes,
            cv=cv,
            scoring="accuracy",
            n_jobs=1,
        )

        # learning_curve returns shapes (n_ticks, n_splits)
        # Some fits can fail for very small training sizes leading to nan scores.
        # Use nan-aware statistics and replace any remaining nans with 0.0 so outputs are numeric in [0,1].
        train_mean = np.nanmean(train_scores, axis=1)
        train_std = np.nanstd(train_scores, axis=1)
        valid_mean = np.nanmean(valid_scores, axis=1)
        valid_std = np.nanstd(valid_scores, axis=1)

        # Replace NaNs (e.g. if all folds failed for a size) with 0.0
        train_mean = np.where(np.isnan(train_mean), 0.0, train_mean)
        train_std = np.where(np.isnan(train_std), 0.0, train_std)
        valid_mean = np.where(np.isnan(valid_mean), 0.0, valid_mean)
        valid_std = np.where(np.isnan(valid_std), 0.0, valid_std)

        results[name] = {
            "train_sizes_abs": train_sizes_abs.tolist(),
            "train_score_mean": train_mean.tolist(),
            "train_score_std": train_std.tolist(),
            "valid_score_mean": valid_mean.tolist(),
            "valid_score_std": valid_std.tolist(),
        }

    # Compare final (largest training size) validation means
    final_log = results["logistic"]["valid_score_mean"][-1]
    final_rf = results["random_forest"]["valid_score_mean"][-1]

    best_model = "random_forest" if final_rf >= final_log else "logistic"
    best_valid_mean = max(final_rf, final_log)

    # Simple heuristic recommendation
    gap = results[best_model]["train_score_mean"][-1] - results[best_model]["valid_score_mean"][-1]
    if gap > 0.08:
        recommendation = "Generalisation gap is notable; collecting more labelled data may help."
    elif gap > 0.02:
        recommendation = "Small generalisation gap; modest gains possible from more data or regularisation."
    else:
        recommendation = "Low generalisation gap; adding data is less likely to yield large gains."

    report = {
        "dataset": "wine",
        "n_samples": int(n_samples),
        "n_features": int(X.shape[1]),
        "n_classes": int(len(np.unique(y))),
        "cv_folds": int(cv_folds),
        "train_sizes_fractions": train_sizes.tolist(),
        "models": results,
        "best_model": best_model,
        "best_valid_mean": float(best_valid_mean),
        "generalisation_gap": float(gap),
        "recommendation": recommendation,
    }

    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
