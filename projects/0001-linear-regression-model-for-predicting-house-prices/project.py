import json
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


def _make_synthetic_housing(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    area = rng.uniform(30, 300, size=n_samples)
    age = rng.randint(0, 81, size=n_samples)
    location = rng.choice([0, 1, 2], size=n_samples, p=[0.3, 0.5, 0.2])
    prop_type = rng.choice([0, 1], size=n_samples, p=[0.6, 0.4])

    base_per_sqm = 3000.0
    loc_premium = np.array([1.5, 1.0, 0.7])[location]
    type_premium = np.where(prop_type == 1, 1.2, 1.0)
    age_depr = 500.0 * age

    price = area * base_per_sqm * loc_premium * type_premium - age_depr
    noise = rng.normal(loc=0.0, scale=10000.0, size=n_samples)
    price = np.maximum(price + noise, 10000.0)

    return pd.DataFrame({
        "area": area,
        "age": age,
        "location": location,
        "prop_type": prop_type,
        "price": price,
    })


def _build_features(df: pd.DataFrame) -> pd.DataFrame:
    combo = df["location"].astype(str) + "_" + df["prop_type"].astype(str)
    X = pd.DataFrame({"area": df["area"], "age": df["age"]})
    dummies = pd.get_dummies(combo, prefix="combo", drop_first=True)
    X = pd.concat([X, dummies], axis=1)
    for col in dummies.columns:
        X[f"area_{col}"] = df["area"] * dummies[col]
    return X.astype(float)


def run(seed: int = 42) -> Dict[str, float]:
    """Train a linear regression to predict price and return test metrics."""
    df = _make_synthetic_housing(n_samples=1200, seed=seed)
    X = _build_features(df)
    y = df["price"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_mae = float(mean_absolute_error(y_test, y_pred))
    test_rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    test_r2 = float(r2_score(y_test, y_pred))

    median_pred = np.median(y_train)
    baseline_preds = np.full_like(y_test, fill_value=median_pred, dtype=float)
    baseline_mae = float(mean_absolute_error(y_test, baseline_preds))

    return {
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "baseline_mae": baseline_mae,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
