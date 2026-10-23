import json
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split


def build_synthetic_transactions(n_samples: int = 500, seed: int = 42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=4,
        n_redundant=0,
        n_clusters_per_class=1,
        weights=[0.93, 0.07],
        class_sep=2.0,
        flip_y=0.0,
        random_state=seed,
    )
    cols = ['f_amount', 'f_freq', 'f_age', 'f_velocity', 'f_geo_score']
    df = pd.DataFrame(X, columns=cols)
    df['id'] = [f'tx_{i:04d}' for i in range(len(df))]
    df['verified_fraud'] = y.astype(int)
    df['f_amount'] = df['f_amount'] - df['f_amount'].min() + 0.1
    return df


def simple_review_rules(df: pd.DataFrame, seed: int = 42) -> pd.Series:
    thr_amount = df['f_amount'].quantile(0.95)
    r_amount = df['f_amount'] > thr_amount
    thr_velocity = df['f_velocity'].quantile(0.95)
    r_velocity = df['f_velocity'] > thr_velocity
    r_combo = (df['f_age'] < df['f_age'].quantile(0.1)) & (df['f_freq'] > df['f_freq'].quantile(0.9))
    flags = pd.DataFrame({'amount': r_amount, 'velocity': r_velocity, 'combo': r_combo})
    return flags.sum(axis=1)


def build_review_queue(df: pd.DataFrame, score: np.ndarray, capacity: int) -> pd.DataFrame:
    dfq = df.copy().assign(score=score)
    dfq = dfq.sort_values('score', ascending=False).head(capacity)
    notes = []
    for _, row in dfq.iterrows():
        parts = []
        if row['f_amount'] >= df['f_amount'].quantile(0.95):
            parts.append('high_amount')
        if row['f_velocity'] >= df['f_velocity'].quantile(0.95):
            parts.append('high_velocity')
        if row['f_age'] <= df['f_age'].quantile(0.1):
            parts.append('new_account')
        notes.append(','.join(parts) if parts else 'none_obvious')
    return dfq.assign(audit_note=notes)


def run(seed: int = 42) -> Dict[str, Any]:
    np.random.seed(seed)
    df = build_synthetic_transactions(n_samples=500, seed=seed)
    train, hold = train_test_split(df, test_size=0.4, random_state=seed, stratify=df['verified_fraud'])

    feature_cols = [c for c in df.columns if c.startswith('f_')]

    clf = RandomForestClassifier(
        n_estimators=300,
        min_samples_leaf=2,
        class_weight='balanced_subsample',
        random_state=seed,
        n_jobs=1,
    )
    clf.fit(train[feature_cols], train['verified_fraud'])
    hold_scores_model = clf.predict_proba(hold[feature_cols])[:, 1]

    iso = IsolationForest(n_estimators=200, contamination=0.07, random_state=seed)
    iso.fit(train[feature_cols])
    hold_scores_iso = -iso.decision_function(hold[feature_cols])

    hold_rules = simple_review_rules(hold, seed=seed).values
    capacity = 8

    idx_iso = np.argsort(-hold_scores_iso)[:capacity]
    iso_top = hold.iloc[idx_iso]
    iso_precision = iso_top['verified_fraud'].mean()
    iso_recall = iso_top['verified_fraud'].sum() / max(1, hold['verified_fraud'].sum())

    rule_order = np.lexsort((-hold['f_amount'].values, -hold_rules))
    rule_top = hold.iloc[rule_order][:capacity]
    rule_precision = rule_top['verified_fraud'].mean()
    rule_recall = rule_top['verified_fraud'].sum() / max(1, hold['verified_fraud'].sum())

    def norm(x):
        x = np.asarray(x, dtype=float)
        if x.max() == x.min():
            return np.zeros_like(x)
        return (x - x.min()) / (x.max() - x.min())

    combined_score = 0.85 * norm(hold_scores_model) + 0.10 * norm(hold_rules) + 0.05 * norm(hold_scores_iso)
    combined_queue = build_review_queue(hold.reset_index(drop=True), combined_score, capacity)
    combined_precision = combined_queue['verified_fraud'].mean()
    combined_recall = combined_queue['verified_fraud'].sum() / max(1, hold['verified_fraud'].sum())

    missed = hold[~hold['id'].isin(combined_queue['id']) & (hold['verified_fraud'] == 1)]

    return {
        'precision_at_capacity': float(round(combined_precision, 4)),
        'recall_at_capacity': float(round(combined_recall, 4)),
        'isolation_precision': float(round(iso_precision, 4)),
        'rule_precision': float(round(rule_precision, 4)),
        'review_capacity': int(capacity),
        'n_verified_in_queue': int(combined_queue['verified_fraud'].sum()),
        'n_missed_verified': int(missed.shape[0]),
        'dataset_size': int(len(df)),
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
