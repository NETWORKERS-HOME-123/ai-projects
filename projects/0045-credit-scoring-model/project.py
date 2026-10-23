import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, brier_score_loss


def generate_synthetic_credit(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    age = rng.normal(40, 10, size=n_samples).clip(18, 85)
    income = np.exp(rng.normal(10.5, 0.8, size=n_samples))  # positive skew
    employment_years = np.abs(rng.normal(5, 3, size=n_samples)).clip(0, 40)
    credit_history = np.abs(rng.normal(8, 6, size=n_samples)).clip(0, 40)
    prev_defaults = rng.binomial(1, 0.08, size=n_samples)

    # sensitive attribute: region A/B (simulates group differences)
    region = rng.binomial(1, 0.45, size=n_samples)  # 0 or 1

    # proxy variable correlated with region (e.g., certain postcode-derived score)
    proxy = region * rng.normal(1.2, 0.3, size=n_samples) + (1 - region) * rng.normal(0.9, 0.3, size=n_samples)

    # Create a synthetic probability of default using a stronger logistic signal
    # Increased coefficients give a clearer separation so that a simple model can achieve a reasonable AUC
    logit = (
        -3.0
        + 0.02 * (age - 40)
        - 0.00005 * (income - income.mean())
        - 0.18 * employment_years
        - 0.12 * credit_history
        + 1.8 * prev_defaults
        + 0.6 * proxy
    )
    prob_default = 1 / (1 + np.exp(-logit))
    default = (rng.rand(n_samples) < prob_default).astype(int)

    df = pd.DataFrame(
        {
            "age": age,
            "income": income,
            "employment_years": employment_years,
            "credit_history": credit_history,
            "prev_defaults": prev_defaults,
            "proxy": proxy,
            "region": region,
            "default": default,
        }
    )

    # Introduce missingness: missing at random in income and employment
    miss_idx = rng.choice(n_samples, size=int(0.06 * n_samples), replace=False)
    df.loc[miss_idx, "income"] = np.nan
    miss_idx2 = rng.choice(n_samples, size=int(0.05 * n_samples), replace=False)
    df.loc[miss_idx2, "employment_years"] = np.nan

    return df


def build_features(df: pd.DataFrame) -> (pd.DataFrame, pd.Series):
    X = df[["age", "income", "employment_years", "credit_history", "prev_defaults", "proxy", "region"]].copy()
    y = df["default"].copy()
    return X, y


def run(seed: int = 42) -> Dict:
    # 1. Data
    df = generate_synthetic_credit(n_samples=2000, seed=seed)
    X, y = build_features(df)

    # 2. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # 3. Preprocessing: impute then scale. Keep 'region' as a protected feature (no encoding needed)
    numeric_cols = [c for c in X.columns if c != "region"]
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()

    X_train_num = imputer.fit_transform(X_train[numeric_cols])
    X_train_num = scaler.fit_transform(X_train_num)
    X_train_proc = pd.DataFrame(X_train_num, columns=numeric_cols, index=X_train.index)
    X_train_proc["region"] = X_train["region"].values

    X_test_num = imputer.transform(X_test[numeric_cols])
    X_test_num = scaler.transform(X_test_num)
    X_test_proc = pd.DataFrame(X_test_num, columns=numeric_cols, index=X_test.index)
    X_test_proc["region"] = X_test["region"].values

    # 4. Fit an interpretable baseline model: logistic regression with limited regularisation
    # Increase regularization strength (C) and iterations to ensure stable fitting and better AUC on this synthetic data
    model = LogisticRegression(random_state=seed, solver="liblinear", C=5.0, max_iter=200)
    model.fit(X_train_proc.drop(columns=["region"]), y_train)

    # 5. Predict probabilities and compute metrics
    prob_test = model.predict_proba(X_test_proc.drop(columns=["region"]))[:, 1]
    auc = float(roc_auc_score(y_test, prob_test))
    brier = float(brier_score_loss(y_test, prob_test))

    # 6. Subgroup analysis: false positive rate by region at threshold 0.5
    thresh = 0.5
    pred_label = (prob_test >= thresh).astype(int)
    test_df = X_test_proc.copy()
    test_df["y_true"] = y_test.values
    test_df["y_pred"] = pred_label

    fpr_by_region = {}
    for g in sorted(test_df["region"].unique()):
        group = test_df[test_df["region"] == g]
        negatives = group[group["y_true"] == 0]
        if len(negatives) == 0:
            fpr = 0.0
        else:
            fpr = float((negatives["y_pred"] == 1).mean())
        fpr_by_region[f"fpr_region_{int(g)}"] = fpr

    # 7. Model card: short educational text (no decisions)
    model_card = (
        "Model card - educational baseline. Model: LogisticRegression. "
        "Data: synthetic classroom data. This is not suitable for operational lending. "
        "Appeal: any adverse outcome described here would require human review, documented reason codes and an appeal process. "
        "Reviewers should check feature provenance, missing data handling and proxy correlations (for example the 'proxy' column)."
    )

    metrics = {
        "auc": auc,
        "brier": brier,
        "threshold": float(thresh),
        "fpr_region_0": fpr_by_region["fpr_region_0"],
        "fpr_region_1": fpr_by_region["fpr_region_1"],
        "model_card": model_card,
    }

    return metrics


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
