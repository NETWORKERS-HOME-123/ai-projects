import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve


def _make_synthetic_churn(seed: int = 42, n: int = 2000) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    # Time-bounded features: tenure (days), last activity recency (days), recent sessions, avg session length
    tenure = rng.exponential(scale=400, size=n).astype(int) + 1
    recency = rng.exponential(scale=40, size=n).astype(int)
    recent_sessions = rng.poisson(lam=2 + (tenure / 400), size=n)
    avg_session_length = rng.normal(loc=10 + (recent_sessions * 0.5), scale=3.0, size=n)
    # Flags: opted_out_of_marketing, has_subscription
    opted_out = rng.binomial(1, 0.15, size=n)
    has_subscription = (rng.rand(n) < (0.5 + (tenure / (tenure.max() * 2)))).astype(int)

    # Construct a churn probability that depends on recency, low usage, short tenure, and lack of subscription.
    # Ensure this is fully computed from features available at the observation window.
    logit = (
        -0.02 * tenure
        + 0.08 * recency
        - 0.5 * has_subscription
        - 0.15 * recent_sessions
        - 0.05 * avg_session_length
        + 0.6 * (opted_out)
    )
    # Add some noise
    logit += rng.normal(scale=0.8, size=n)
    prob = 1 / (1 + np.exp(-logit))

    churn = rng.binomial(1, prob, size=n)

    df = pd.DataFrame(
        {
            "tenure": tenure,
            "recency": recency,
            "recent_sessions": recent_sessions,
            "avg_session_length": avg_session_length,
            "opted_out": opted_out,
            "has_subscription": has_subscription,
            "churn": churn,
            "churn_prob_sim": prob,
        }
    )
    return df


def _recency_rule_score(recency: np.ndarray) -> np.ndarray:
    # A simple score inversely related to recency: recent activity -> lower churn score
    # We map recency to a [0,1] score where higher means more likely to churn
    return (recency - recency.min()) / (recency.max() - recency.min() + 1e-9)


def run(seed: int = 42) -> Dict[str, Any]:
    # Build synthetic time-bounded dataset (teaching example)
    df = _make_synthetic_churn(seed=seed, n=2000)

    features = ["tenure", "recency", "recent_sessions", "avg_session_length", "opted_out", "has_subscription"]
    X = df[features].values
    y = df["churn"].values

    # Use a later-account holdout split: simulate train on earlier subset by stratifying on tenure quantile
    # For reproducibility use standard train_test_split with fixed random_state
    X_train, X_test, y_train, y_test, df_train, df_test = train_test_split(
        X, y, df, test_size=0.25, random_state=seed, stratify=(df["churn"])
    )

    # Baseline: calibrated logistic regression
    base = LogisticRegression(max_iter=1000, random_state=seed)
    calib = CalibratedClassifierCV(base, cv=5)
    calib.fit(X_train, y_train)
    prob_pos = calib.predict_proba(X_test)[:, 1]

    # Recency rule
    recency_scores = _recency_rule_score(df_test["recency"].values)

    # Evaluation: average precision (AP) and Brier score for calibration
    ap = float(average_precision_score(y_test, prob_pos))
    brier = float(brier_score_loss(y_test, prob_pos))
    ap_recency = float(average_precision_score(y_test, recency_scores))

    # Precision at recall >= 0.5 (if available)
    precisions, recalls, thresholds = precision_recall_curve(y_test, prob_pos)
    # Find precision at recall >= 0.5; if none, take best precision
    mask = recalls >= 0.5
    if mask.any():
        precision_at_recall_0_5 = float(precisions[mask].max())
    else:
        precision_at_recall_0_5 = float(precisions.max())

    # Build a small staff-reviewed support queue: top predicted churners with reasons and opt-outs
    df_test = df_test.copy().reset_index(drop=True)
    df_test["predicted_prob"] = prob_pos
    queue = df_test.sort_values("predicted_prob", ascending=False).head(8)
    # For each queued account provide a compact reason string and an opt-out flag
    review_rows = []
    for _, row in queue.iterrows():
        reasons = []
        if row["recency"] > 30:
            reasons.append("long_recency")
        if row["recent_sessions"] <= 1:
            reasons.append("low_activity")
        if row["opted_out"]:
            reasons.append("opted_out_of_marketing")
        if not row["has_subscription"]:
            reasons.append("no_subscription")
        review_rows.append(
            {
                "id": int(row.name),
                "predicted_prob": float(row["predicted_prob"]),
                "reasons": ",".join(reasons) if reasons else "none",
                "opted_out": bool(row["opted_out"]),
            }
        )

    result = {
        "average_precision": ap,
        "brier_score": brier,
        "recency_rule_ap": ap_recency,
        "precision_at_recall_0.5": precision_at_recall_0_5,
        "review_queue_sample": review_rows,
        "n_test": int(len(y_test)),
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
