import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)

    n = 3000
    X, y_raw = make_regression(
        n_samples=n,
        n_features=10,
        n_informative=10,
        noise=2.0,
        random_state=seed,
    )
    y_shifted = y_raw - y_raw.min() + 1.0
    base_revenue = 50.0
    y = base_revenue * y_shifted

    signup_month = rng.randint(0, 12, size=n)
    month_drift = (signup_month - signup_month.mean()) * rng.uniform(0.02, 0.05)
    y = y * (1.0 + month_drift)

    censor_mask = rng.rand(n) < 0.08
    censor_factor = rng.uniform(0.3, 0.8, size=n)
    y = y * np.where(censor_mask, censor_factor, 1.0)
    y = np.maximum(0.0, y)

    feature_cols = [f"feat_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=feature_cols)
    df["signup_month"] = signup_month
    df["target_revenue"] = y

    known_features = feature_cols + ["signup_month"]

    train_df, test_df = train_test_split(df, test_size=0.25, random_state=seed)

    group_means = train_df.groupby("signup_month")["target_revenue"].mean().to_dict()
    global_mean = train_df["target_revenue"].mean()

    def group_predict(rows: pd.DataFrame) -> np.ndarray:
        return rows["signup_month"].map(lambda c: group_means.get(int(c), global_mean)).values

    y_test = test_df["target_revenue"].values
    y_baseline_pred = group_predict(test_df)
    mae_baseline = float(mean_absolute_error(y_test, y_baseline_pred))

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            ("rf", RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)),
        ]
    )
    pipeline.fit(train_df[known_features], train_df["target_revenue"])
    y_pred = pipeline.predict(test_df[known_features])

    mae_model = float(mean_absolute_error(y_test, y_pred))
    r2 = float(r2_score(y_test, y_pred))

    group_maes = {}
    for g in sorted(test_df["signup_month"].unique()):
        mask = test_df["signup_month"] == g
        group_maes[int(g)] = float(mean_absolute_error(y_test[mask], y_pred[mask]))
    group_mae_mean = float(np.mean(list(group_maes.values())))

    terciles = np.percentile(train_df["target_revenue"], [33.33, 66.66])
    bands = {"low": (None, terciles[0]), "mid": (terciles[0], terciles[1]), "high": (terciles[1], None)}
    band_maes = {}
    for name, (lo, hi) in bands.items():
        if lo is None:
            mask = test_df["target_revenue"] <= hi
        elif hi is None:
            mask = test_df["target_revenue"] > lo
        else:
            mask = (test_df["target_revenue"] > lo) & (test_df["target_revenue"] <= hi)
        band_maes[name] = float(mean_absolute_error(y_test[mask], y_pred[mask])) if mask.sum() else None

    def aggregate_and_uncertainty(predictions: np.ndarray):
        total = float(np.sum(predictions))
        se = float(np.std(predictions, ddof=1) / np.sqrt(len(predictions))) if len(predictions) > 1 else 0.0
        return total, se * np.sqrt(len(predictions))

    total_pred, total_se = aggregate_and_uncertainty(y_pred)

    return {
        "r2": r2,
        "mae_model": mae_model,
        "mae_baseline_signup_group": mae_baseline,
        "group_mae_mean": group_mae_mean,
        "band_mae_low": band_maes["low"],
        "band_mae_mid": band_maes["mid"],
        "band_mae_high": band_maes["high"],
        "planning_total_pred": total_pred,
        "planning_total_uncertainty_se": total_se,
        "n_test": int(len(test_df)),
        "model": "RandomForestRegressor",
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
