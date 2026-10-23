import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


def _wilson_ci(p: float, n: int, z: float = 1.96):
    if n == 0:
        return 0.0, 1.0
    center = (p + (z * z) / (2 * n)) / (1 + (z * z) / n)
    half = (z * (p * (1 - p) / n + (z * z) / (4 * n * n)) ** 0.5) / (1 + (z * z) / n)
    return max(0.0, center - half), min(1.0, center + half)


def _ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    # Expected Calibration Error (absolute difference, weighted)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if not mask.any():
            continue
        prop = mask.mean()
        avg_prob = y_prob[mask].mean()
        avg_true = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(avg_prob - avg_true)
    return float(ece)


def run(seed: int = 42) -> Dict[str, float]:
    """Train an interpretable baseline and return held-out metrics.

    Returns a JSON-serialisable dict with keys:
      - accuracy, sensitivity, specificity (point estimates)
      - sensitivity_ci, specificity_ci (tuples as two floats in a list)
      - brier, ece (calibration metrics)
      - subgroup_sensitivity_diff (difference between high/low subgroup)
    """
    # Load de-identified per-patient teaching dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)  # 0=malignant,1=benign in this dataset; we'll treat 0 as positive (disease)

    # For clarity, define disease label as 1 for malignant
    y = (y == 0).astype(int)

    # Introduce explicit missingness for teaching: randomly set ~5% values to NaN (reproducible)
    rng = np.random.RandomState(seed)
    X_missing = X.copy()
    # operate on a writable numpy copy to avoid read-only assignment errors
    arr = X_missing.to_numpy().copy()
    mask = rng.rand(*arr.shape) < 0.05
    arr[mask] = np.nan
    # put back into a DataFrame (preserve column names and index)
    X_missing = pd.DataFrame(arr, columns=X.columns, index=X.index)

    # Patient-level split: each row is a patient here; still use stratified split to keep class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X_missing, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Pipeline: explicit per-feature imputation, scaling, and an interpretable logistic baseline
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(solver="liblinear", random_state=seed))
    ])

    pipe.fit(X_train, y_train)

    prob_pos = pipe.predict_proba(X_test)[:, 1]
    pred = (prob_pos >= 0.5).astype(int)

    # Confusion matrix elements
    tp = int(((pred == 1) & (y_test == 1)).sum())
    tn = int(((pred == 0) & (y_test == 0)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())

    # Compute metrics
    acc = float((pred == y_test).mean())
    sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    sens_ci = _wilson_ci(sensitivity, tp + fn)
    spec_ci = _wilson_ci(specificity, tn + fp)

    brier = float(brier_score_loss(y_test, prob_pos))
    ece = float(_ece(y_test.to_numpy(), prob_pos, n_bins=10))

    # Simple subgroup check: split by median of mean radius (a clinically meaningful-size proxy)
    if "mean radius" in X.columns:
        median_radius = X_train["mean radius"].median()
        subgroup_high = X_test["mean radius"] >= median_radius
        subgroup_low = X_test["mean radius"] < median_radius
        def sens_for(mask):
            if mask.sum() == 0:
                return 0.0
            p = pred[mask]
            t = y_test[mask]
            tp_s = int(((p == 1) & (t == 1)).sum())
            fn_s = int(((p == 0) & (t == 1)).sum())
            return float(tp_s / (tp_s + fn_s)) if (tp_s + fn_s) > 0 else 0.0
        sens_high = sens_for(subgroup_high)
        sens_low = sens_for(subgroup_low)
        subgroup_diff = float(sens_high - sens_low)
    else:
        subgroup_diff = 0.0

    return {
        "accuracy": round(acc, 4),
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "sensitivity_ci": [round(sens_ci[0], 4), round(sens_ci[1], 4)],
        "specificity_ci": [round(spec_ci[0], 4), round(spec_ci[1], 4)],
        "brier": round(brier, 4),
        "ece": round(ece, 4),
        "subgroup_sensitivity_diff": round(subgroup_diff, 4),
        "model": "LogisticRegression"
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
