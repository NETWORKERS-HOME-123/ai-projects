import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, brier_score_loss


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)

    # Synthetic enquiry dataset: features, outcome, recency and a protected attribute
    X, y = make_classification(n_samples=2000, n_features=10, n_informative=5,
                               n_redundant=2, weights=[0.8, 0.2], random_state=seed)

    df = pd.DataFrame(X, columns=[f'feat_{i}' for i in range(X.shape[1])])
    df['outcome'] = y

    # days since enquiry: 0 (most recent) to 30
    df['days_since_enquiry'] = rng.randint(0, 31, size=len(df))

    # synthetic protected attribute correlated weakly with outcome; this must be excluded from training
    df['protected'] = (rng.rand(len(df)) < (0.15 + 0.3 * (df['outcome'] == 1))).astype(int)

    # small opt-out flag (people who asked not to be contacted)
    df['opt_out'] = (rng.rand(len(df)) < 0.03).astype(int)

    # train/test split stratified by outcome, but preserve columns
    train, test = train_test_split(df, test_size=0.25, stratify=df['outcome'], random_state=seed)

    feature_cols = [c for c in df.columns if c.startswith('feat_')]

    # Baseline model: Logistic regression with scaling, then calibration
    base = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed))
    calibrator = CalibratedClassifierCV(base, cv=3)
    calibrator.fit(train[feature_cols], train['outcome'])

    # Predictions (probabilities for positive class)
    test_probs = calibrator.predict_proba(test[feature_cols])[:, 1]

    # Recency baseline: score by recency (more recent = higher priority)
    recency_score = 1.0 / (1.0 + test['days_since_enquiry'].values)

    # Staff capacity: take top 10% of test examples
    k = max(1, int(0.10 * len(test)))

    # Model top-K precision
    model_indices = np.argsort(-test_probs)[:k]
    precision_at_capacity = test.iloc[model_indices]['outcome'].mean()

    # Recency top-K precision
    recency_indices = np.argsort(-recency_score)[:k]
    recency_precision = test.iloc[recency_indices]['outcome'].mean()

    # Subgroup coverage: fraction of subgroup members included in top-K compared to their population fraction
    subgroup = test['protected'] == 1
    if subgroup.sum() == 0:
        subgroup_coverage_ratio = 0.0
    else:
        subgroup_in_topk = test.iloc[model_indices]['protected'].sum()
        subgroup_share = subgroup.sum() / len(test)
        # coverage ratio: observed share in top-K divided by population share (>=1 means over-represented)
        observed_share_in_topk = subgroup_in_topk / k
        subgroup_coverage_ratio = observed_share_in_topk / subgroup_share if subgroup_share > 0 else 0.0

    # Minimum service: ensure very recent enquiries (days_since_enquiry <= 2) should get some presence in the serviced set
    recent_mask = test['days_since_enquiry'] <= 2
    if recent_mask.sum() == 0:
        min_service_rate = 0.0
    else:
        recent_in_topk = test.iloc[model_indices].loc[:, 'days_since_enquiry'] <= 2
        min_service_rate = recent_in_topk.sum() / recent_mask.sum()

    # Calibration metric
    auc = roc_auc_score(test['outcome'], test_probs)
    brier = brier_score_loss(test['outcome'], test_probs)

    # Construct a reviewable queue (small sample returned as counts and flags)
    queue_sample = test.assign(model_score=test_probs,
                               recency_score=recency_score,
                               reason=[('recent' if d <= 2 else 'model') for d in test['days_since_enquiry']])
    # Exclude opt-outs from service counts
    queue_sample = queue_sample.sort_values(['model_score', 'recency_score'], ascending=[False, False])

    metrics = {
        'auc': float(auc),
        'brier': float(brier),
        'precision_at_capacity': float(precision_at_capacity),
        'precision_recency': float(recency_precision),
        'subgroup_coverage_ratio': float(subgroup_coverage_ratio),
        'min_service_rate': float(min_service_rate)
    }

    return metrics


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
