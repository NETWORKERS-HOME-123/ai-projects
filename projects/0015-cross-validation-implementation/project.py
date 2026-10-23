import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run(seed: int = 42) -> Dict:
    """Run a stratified cross-validation experiment with a holdout.

    Returns a JSON-serialisable dict with CV statistics, holdout accuracy,
    number of folds and a fold report describing indices and preprocessing
    boundaries per fold. All randomness is controlled with the `seed`.
    """
    rng = np.random.RandomState(seed)

    # 1) Prepare synthetic data (students should replace with real data)
    X, y = make_classification(
        n_samples=800,
        n_features=10,
        n_informative=4,
        n_redundant=1,
        n_clusters_per_class=2,
        weights=[0.6, 0.4],
        flip_y=0.02,
        class_sep=1.0,
        random_state=seed,
    )

    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y
    df.index = list(range(len(df)))  # explicit integer index for reporting

    # 2) Create a holdout (stratified) and keep it untouched during CV
    train_df, holdout_df = train_test_split(
        df, test_size=0.20, stratify=df["y"], random_state=seed
    )
    train_df = train_df.reset_index(drop=False).rename(columns={"index": "orig_index"})
    holdout_df = holdout_df.reset_index(drop=False).rename(columns={"index": "orig_index"})

    X_train = train_df[[c for c in train_df.columns if c.startswith("x")]].values
    y_train = train_df["y"].values
    X_hold = holdout_df[[c for c in holdout_df.columns if c.startswith("x")]].values
    y_hold = holdout_df["y"].values

    # 3) Build pipeline (imputer + scaler + classifier)
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(solver="liblinear", random_state=seed),
            ),
        ]
    )

    # 4) Run stratified K-fold cross-validation manually to build a fold report
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    fold_scores: List[float] = []
    fold_report: List[Dict] = []

    for fold_id, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        X_tr, y_tr = X_train[train_idx], y_train[train_idx]
        X_val, y_val = X_train[val_idx], y_train[val_idx]

        # Fit pipeline on this fold's training data
        pipeline.fit(X_tr, y_tr)

        # Measure accuracy on validation fold
        preds = pipeline.predict(X_val)
        acc = accuracy_score(y_val, preds)
        fold_scores.append(float(acc))

        # Capture preprocessing boundaries: mean and std computed on training features
        # after imputation but before classifier (i.e. scaler parameters)
        scaler = pipeline.named_steps["scaler"]
        # scaler.mean_ and scaler.scale_ are numpy arrays
        train_mean = scaler.mean_.tolist()
        train_scale = scaler.scale_.tolist()

        # Record original indices for transparency (these point into the original df index)
        orig_train_idx = train_df.loc[train_idx, "orig_index"].astype(int).tolist()
        orig_val_idx = train_df.loc[val_idx, "orig_index"].astype(int).tolist()

        fold_report.append(
            {
                "fold": int(fold_id),
                "train_orig_indices": orig_train_idx,
                "val_orig_indices": orig_val_idx,
                "train_scaler_mean": train_mean,
                "train_scaler_scale": train_scale,
                "val_accuracy": float(acc),
            }
        )

    # 5) Summarise CV results
    cv_mean = float(np.mean(fold_scores))
    cv_std = float(np.std(fold_scores, ddof=1))

    # 6) Final evaluation: fit on full training set and evaluate on holdout
    pipeline.fit(X_train, y_train)
    hold_preds = pipeline.predict(X_hold)
    holdout_accuracy = float(accuracy_score(y_hold, hold_preds))

    results = {
        "cv_mean_accuracy": cv_mean,
        "cv_std_accuracy": cv_std,
        "holdout_accuracy": holdout_accuracy,
        "n_folds": int(n_splits),
        "fold_report": fold_report,
    }

    return results


if __name__ == "__main__":
    import sys
    print(json.dumps(run(seed=42), indent=2))
