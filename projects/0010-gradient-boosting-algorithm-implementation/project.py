import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, brier_score_loss


def _build_synthetic_subscription_data(n_samples: int = 2000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    # Features that plausibly relate to churn
    tenure_days = rng.exponential(scale=200, size=n_samples).astype(int)
    avg_weekly_usage = rng.normal(loc=5, scale=2, size=n_samples)
    support_tickets = rng.poisson(lam=0.2, size=n_samples)
    plan_tier = rng.choice([0, 1, 2], size=n_samples, p=[0.6, 0.3, 0.1])  # 0=basic,1=pro,2=premium
    recent_login_gap = rng.exponential(scale=10, size=n_samples)

    # A clear, time-based outcome: churn within 90 days of the snapshot is deterministic-ish
    # Use an interpretable rule plus noise — this mimics a permissioned historical label.
    churn_prob_clean = (
        (tenure_days < 90).astype(float) * 0.45
        + (avg_weekly_usage < 3).astype(float) * 0.25
        + (support_tickets > 1).astype(float) * 0.15
        + (recent_login_gap > 14).astype(float) * 0.15
        - plan_tier * 0.10
    )
    churn_prob = np.clip(churn_prob_clean + rng.normal(scale=0.05, size=n_samples), 0, 1)
    churn = rng.binomial(1, churn_prob)

    df = pd.DataFrame(
        {
            "tenure_days": tenure_days,
            "avg_weekly_usage": avg_weekly_usage,
            "support_tickets": support_tickets,
            "plan_tier": plan_tier,
            "recent_login_gap": recent_login_gap,
            "churn": churn,
        }
    )
    return df


def run(seed: int = 42) -> Dict:
    """Train a GradientBoostingClassifier to predict churn on synthetic subscription data.

    Returns a dict of JSON-serialisable metrics. Deterministic for a fixed seed.
    """
    data = _build_synthetic_subscription_data(n_samples=2000, seed=seed)
    X = data[["tenure_days", "avg_weekly_usage", "support_tickets", "plan_tier", "recent_login_gap"]]
    y = data["churn"]

    # Train / test split: hold out a test set and never touch it during tuning.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=seed
    )

    # Baseline: predict the most frequent class
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    yb_pred = baseline.predict(X_test)
    baseline_acc = float(accuracy_score(y_test, yb_pred))

    # Simple logistic regression baseline (regularised)
    logreg = LogisticRegression(max_iter=1000, random_state=seed, solver="liblinear")
    logreg.fit(X_train, y_train)
    ylr_pred = logreg.predict(X_test)
    logreg_acc = float(accuracy_score(y_test, ylr_pred))

    # Gradient boosting: tune a very small grid on the training set only
    gbc = GradientBoostingClassifier(random_state=seed)
    param_grid = {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}
    grid = GridSearchCV(gbc, param_grid, cv=3, scoring="precision", n_jobs=1)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_

    # Evaluate on the held-out test set
    y_pred = best.predict(X_test)
    y_proba = best.predict_proba(X_test)[:, 1]

    gb_accuracy = float(accuracy_score(y_test, y_pred))
    gb_precision = float(precision_score(y_test, y_pred, zero_division=0))
    gb_recall = float(recall_score(y_test, y_pred, zero_division=0))
    gb_brier = float(brier_score_loss(y_test, y_proba))

    # Conceptual intervention: score-based opt-in fraction and expected positives among invitees
    # This is a planning metric, not an operational claim. We select top-k by predicted probability.
    invite_fraction = 0.10
    k = max(1, int(len(y_test) * invite_fraction))
    top_idx = np.argsort(-y_proba)[:k]
    invite_precision = float(precision_score(y_test.iloc[top_idx], (y_proba[top_idx] > 0.5).astype(int), zero_division=0))

    results = {
        "baseline_mostfreq_accuracy": baseline_acc,
        "logreg_accuracy": logreg_acc,
        "gbc_best_params": str(grid.best_params_),
        "gbc_accuracy": gb_accuracy,
        "gbc_precision": gb_precision,
        "gbc_recall": gb_recall,
        "gbc_brier_score": gb_brier,
        "invite_fraction": float(invite_fraction),
        "invite_precision": invite_precision,
    }
    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
