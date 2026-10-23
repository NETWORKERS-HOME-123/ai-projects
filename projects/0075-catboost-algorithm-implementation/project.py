import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score


def _build_mixed_dataset(seed: int = 42, n_samples: int = 1200):
    rng = np.random.RandomState(seed)
    X_num, y = make_classification(n_samples=n_samples, n_features=6, n_informative=4,
                                   n_redundant=0, random_state=seed, flip_y=0.03)
    df = pd.DataFrame(X_num, columns=[f"num_{i}" for i in range(X_num.shape[1])])

    # Create categorical features with different cardinalities
    # cat_low: 3 categories, cat_med: 8 categories, cat_high: 40 categories
    df["cat_low"] = rng.choice(["red", "green", "blue"], size=n_samples, p=[0.5, 0.3, 0.2])
    df["cat_med"] = rng.choice([f"m{i}" for i in range(8)], size=n_samples)
    df["cat_high"] = rng.choice([f"h{i}" for i in range(40)], size=n_samples)

    # Introduce missing values at random in categorical columns
    for col in ["cat_low", "cat_med", "cat_high"]:
        mask = rng.rand(n_samples) < 0.06
        df.loc[mask, col] = None

    # Introduce some numeric missingness
    for col in [c for c in df.columns if c.startswith("num_")]:
        mask = rng.rand(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    # Attach target
    df["target"] = y
    return df


def run(seed: int = 42) -> Dict:
    # Build data
    df = _build_mixed_dataset(seed=seed, n_samples=1200)
    cat_cols = ["cat_low", "cat_med", "cat_high"]
    num_cols = [c for c in df.columns if c.startswith("num_")]

    # Split into train / val / test (stratified)
    trainval, test = train_test_split(df, test_size=0.2, stratify=df["target"], random_state=seed)
    train, val = train_test_split(trainval, test_size=0.25, stratify=trainval["target"], random_state=seed)

    # Inject unseen categories into the test set to simulate deployment surprises
    rng = np.random.RandomState(seed)
    test = test.copy()
    # add a new unseen level to cat_med and cat_high for 8% of rows
    unseen_mask = rng.rand(len(test)) < 0.08
    test.loc[unseen_mask, "cat_med"] = "unseen_med"
    test.loc[unseen_mask, "cat_high"] = "unseen_high"

    X_train = train[num_cols + cat_cols]
    y_train = train["target"].values
    X_val = val[num_cols + cat_cols]
    y_val = val["target"].values
    X_test = test[num_cols + cat_cols]
    y_test = test["target"].values

    # Baseline: One-hot encoding + LogisticRegression
    ohe_pipe = Pipeline([
        ("num_imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    ohe_cat = Pipeline([
        ("cat_imputer", SimpleImputer(strategy="constant", fill_value="__MISSING__")),
        # newer sklearn uses sparse_output, older uses sparse. The test environment expects sparse_output.
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    # Prepare combined features manually (small project, avoid ColumnTransformer for clarity)
    def _prepare_ohe(X_df):
        num = ohe_pipe.fit_transform(X_df[num_cols])
        cat = ohe_cat.fit_transform(X_df[cat_cols])
        return np.hstack([num, cat])

    X_train_ohe = _prepare_ohe(X_train)
    # For validation and test, transform using fitted parts
    num_imputer = ohe_pipe.named_steps["num_imputer"]
    scaler = ohe_pipe.named_steps["scaler"]
    cat_imputer = ohe_cat.named_steps["cat_imputer"]
    ohe = ohe_cat.named_steps["ohe"]

    def _transform_ohe(X_df):
        num = scaler.transform(num_imputer.transform(X_df[num_cols]))
        cat = ohe.transform(cat_imputer.transform(X_df[cat_cols]))
        return np.hstack([num, cat])

    X_val_ohe = _transform_ohe(X_val)
    X_test_ohe = _transform_ohe(X_test)

    clf_ohe = LogisticRegression(max_iter=200, random_state=seed)
    clf_ohe.fit(X_train_ohe, y_train)

    # Emulated CatBoost: ordinal encoding + HistGradientBoostingClassifier
    # Ordinal encoder uses -1 for missing and produces integer categories; unseen categories at transform will be set to a special value.
    ord_imputer = SimpleImputer(strategy="constant", fill_value="__MISSING__")
    ord_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)

    num_imp = SimpleImputer(strategy="mean").fit(X_train[num_cols])
    X_train_num = num_imp.transform(X_train[num_cols])
    X_val_num = num_imp.transform(X_val[num_cols])
    X_test_num = num_imp.transform(X_test[num_cols])

    X_train_cat = ord_imputer.fit_transform(X_train[cat_cols])
    ord_encoder.fit(X_train_cat)
    X_train_cat_enc = ord_encoder.transform(X_train_cat)
    X_val_cat_enc = ord_encoder.transform(ord_imputer.transform(X_val[cat_cols]))
    X_test_cat_enc = ord_encoder.transform(ord_imputer.transform(X_test[cat_cols]))

    X_train_ord = np.hstack([X_train_num, X_train_cat_enc.astype(float)])
    X_val_ord = np.hstack([X_val_num, X_val_cat_enc.astype(float)])
    X_test_ord = np.hstack([X_test_num, X_test_cat_enc.astype(float)])

    gb = HistGradientBoostingClassifier(random_state=seed, max_iter=200)
    gb.fit(X_train_ord, y_train)

    # Evaluate
    def _metrics(clf, X, y_true):
        probs = clf.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)
        return {
            "accuracy": float(accuracy_score(y_true, preds)),
            "roc_auc": float(roc_auc_score(y_true, probs)),
        }

    ohe_test_metrics = _metrics(clf_ohe, X_test_ohe, y_test)
    ord_test_metrics = _metrics(gb, X_test_ord, y_test)

    # Categorical feature schema report
    cat_schema = {
        "cat_low": {"cardinality": 3, "description": "low-cardinality nominal"},
        "cat_med": {"cardinality": 8, "description": "medium-cardinality nominal; unseen category injected in test"},
        "cat_high": {"cardinality": 40, "description": "high-cardinality nominal; unseen category injected in test"},
    }

    report = {
        "dataset_rows": int(len(df)),
        "categorical_schema": cat_schema,
        "onehot_baseline": {
            "accuracy_test": ohe_test_metrics["accuracy"],
            "roc_auc_test": ohe_test_metrics["roc_auc"],
        },
        "ordinal_gb": {
            "accuracy_test": ord_test_metrics["accuracy"],
            "roc_auc_test": ord_test_metrics["roc_auc"],
        },
    }

    # Main metric for tests: ordinal_gb.accuracy_test
    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
