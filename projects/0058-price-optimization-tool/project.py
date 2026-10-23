import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


def _make_data(seed: int, n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    seasonality = np.sin(2 * np.pi * t / 52)
    promo = rng.binomial(1, 0.25, n)
    inventory = rng.integers(80, 180, n)
    price = rng.uniform(70, 130, n)
    cost = np.full(n, 42.0)
    base = 185 + 14 * seasonality + 16 * promo + 0.07 * inventory
    noise = rng.normal(0, 7.5, n)
    demand = np.maximum(1.0, base - 1.35 * price + noise)
    return pd.DataFrame({"price": price, "promo": promo, "inventory": inventory, "seasonality": seasonality, "cost": cost, "demand": demand})


def _score_scenario(model, price: float, base_row: pd.Series) -> dict:
    x = pd.DataFrame([{"price": price, "promo": base_row.promo, "inventory": base_row.inventory, "seasonality": base_row.seasonality}])
    demand = float(max(0.0, model.predict(x)[0]))
    units = min(demand, float(base_row.inventory))
    revenue = price * units
    margin = (price - float(base_row.cost)) * units
    return {"price": round(price, 2), "predicted_demand": round(demand, 3), "revenue": round(revenue, 3), "margin": round(margin, 3)}


def run(seed: int = 42) -> dict:
    data = _make_data(seed)
    train, test = train_test_split(data, test_size=0.25, random_state=seed)
    features = ["price", "promo", "inventory", "seasonality"]
    model = LinearRegression().fit(train[features], train["demand"])
    preds = model.predict(test[features])
    held_out_r2 = float(r2_score(test["demand"], preds))
    held_out_mae = float(mean_absolute_error(test["demand"], preds))
    avg = train[features + ["cost"]].mean(numeric_only=True)
    base_row = pd.Series(avg)
    grid = np.linspace(70, 130, 61)
    scenarios = [_score_scenario(model, p, base_row) for p in grid]
    best_rev = max(scenarios, key=lambda s: s["revenue"])
    best_margin = max(scenarios, key=lambda s: s["margin"])
    base_price = 100.0
    base_s = _score_scenario(model, base_price, base_row)
    elasticity_proxy = float(-model.coef_[0] * base_price / max(base_s["predicted_demand"], 1e-6))
    return {"held_out_r2": round(held_out_r2, 4), "held_out_mae": round(held_out_mae, 4), "elasticity_proxy": round(elasticity_proxy, 4), "baseline_price": round(base_price, 2), "recommended_revenue_price": best_rev["price"], "recommended_margin_price": best_margin["price"], "baseline_revenue": base_s["revenue"], "best_revenue": best_rev["revenue"], "best_margin": best_margin["margin"]}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
