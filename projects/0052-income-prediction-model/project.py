import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score


def run(seed: int = 42) -> Dict:
    """Run a deterministic, small income-prediction experiment.

    The dataset is synthetic (make_regression) and augmented with a synthetic
    sensitive attribute 'gender' (0/1). The target is a continuous amount that
    stands in for income (simulated). Returns a JSON-serialisable dict.
    """
    rng = np.random.RandomState(seed)

    # 1) Create synthetic regression data (labels represent simulated income amounts)
    # Make the underlying regression easier to learn (more informative features,
    # lower noise) so the simple linear baseline achieves a reasonable R^2.
    X, y = make_regression(
        n_samples=2000,
        n_features=8,
        n_informative=8,
        noise=8.0,
        random_state=seed,
    )

    # Convert to a DataFrame for clarity
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])

    # Scale the raw target to be in a plausible 'income' range (thousands)
    # This is purely synthetic: target_mean ~ 50 (k), std ~ 15 (k)
    y = (y - y.mean()) / (y.std() + 1e-12) * 15.0 + 50.0
    df["income_k"] = y

    # 2) Create a synthetic sensitive attribute (gender proxy) correlated weakly with features
    #    Make 0/1 with balanced prevalence and a small group effect on the target
    prob = 0.5
    gender = rng.binomial(1, prob, size=len(df))
    df["gender"] = gender

    # Introduce a small group shift in the target to allow subgroup error analysis
    # This mimics historical bias for teaching purposes only
    df.loc[df["gender"] == 1, "income_k"] += 2.0  # group 1 has +2k mean shift

    # 3) Train / test split (held-out evaluation)
    train, test = train_test_split(df, test_size=0.30, random_state=seed)

    features = [c for c in df.columns if c.startswith("f")]

    # 4) Fit an interpretable linear baseline pipeline
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("linreg", LinearRegression()),
    ])

    pipeline.fit(train[features], train["income_k"])

    # 5) Evaluate on held-out test set
    preds = pipeline.predict(test[features])
    rmse = float(np.sqrt(mean_squared_error(test["income_k"], preds)))
    r2 = float(r2_score(test["income_k"], preds))

    # Subgroup errors: compute RMSE per gender group in the test set
    subgroup_rmse = {}
    for g in sorted(test["gender"].unique()):
        mask = test["gender"] == g
        if mask.sum() == 0:
            subgroup_rmse[f"rmse_gender_{int(g)}"] = None
        else:
            val = float(np.sqrt(mean_squared_error(test.loc[mask, "income_k"], preds[mask.values])))
            subgroup_rmse[f"rmse_gender_{int(g)}"] = val

    # Coefficients summary (model interpretability)
    lin: LinearRegression = pipeline.named_steps["linreg"]
    coef_mean = float(np.mean(np.abs(lin.coef_)))

    report = {
        "n_samples": int(len(df)),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "r2": r2,
        "rmse": rmse,
        "coef_mean_abs": coef_mean,
    }

    # Add subgroup keys
    report.update(subgroup_rmse)

    # Short classroom analysis text (small, reviewable output)
    report["analysis_note"] = (
        "Synthetic regression dataset (make_regression). Target is a simulated\n"
        "continuous income amount in thousands. Model: LinearRegression pipeline.\n"
        "Do not use for decisions about people. See README for provenance and limits."
    )

    return report


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
