import json
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier


def make_mixed_table(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    num_1 = rng.normal(0.0, 1.0, n_samples)
    num_2 = rng.normal(1.0, 2.0, n_samples)
    cats = np.array(['A', 'B', 'C', 'D', 'E'])
    cat = rng.choice(cats, size=n_samples, p=[0.4, 0.3, 0.15, 0.1, 0.05])
    start = np.datetime64('2020-01-01')
    days = rng.randint(0, 365, n_samples)
    dt = start + days.astype('timedelta64[D]')
    future_flag = (rng.rand(n_samples) < (0.25 + 0.45 * (num_1 > 0))).astype(int)
    logits = 1.4 * num_1 + 0.9 * (cat == 'A').astype(int) + 2.0 * future_flag - 0.2
    prob = 1 / (1 + np.exp(-logits))
    y = (rng.rand(n_samples) < prob).astype(int)
    df = pd.DataFrame({
        'num_1': num_1,
        'num_2': num_2,
        'category': cat,
        'event_date': pd.to_datetime(dt),
        'future_flag': future_flag,
        'target': y,
    })
    df.loc[rng.rand(n_samples) < 0.05, 'num_2'] = np.nan
    df.loc[rng.rand(n_samples) < 0.03, 'category'] = np.nan
    return df


class FeatureTransformV1:
    version = 'v2'
    required_columns = ['num_1', 'num_2', 'category', 'future_flag']

    def __init__(self):
        num_cols = ['num_1', 'num_2', 'future_flag']
        cat_cols = ['category']
        num_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler()),
        ])
        try:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        cat_pipe = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
            ('ohe', ohe),
        ])
        self.preprocessor = ColumnTransformer([
            ('num', num_pipe, num_cols),
            ('cat', cat_pipe, cat_cols),
        ])
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        self.pipeline = Pipeline([('pre', self.preprocessor), ('clf', self.model)])

    def check_and_fill(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for c in self.required_columns:
            if c not in X.columns:
                X[c] = np.nan
        return X[self.required_columns]

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        self.pipeline.fit(self.check_and_fill(X), y)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict(self.check_and_fill(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(self.check_and_fill(X))


def baseline_pipeline():
    num_pipe = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
    ])
    pre = ColumnTransformer([('num', num_pipe, ['num_1', 'num_2'])])
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    return Pipeline([('pre', pre), ('clf', clf)])


def run(seed: int = 42) -> Dict:
    df = make_mixed_table(n_samples=1000, seed=seed)
    X = df.drop(columns=['target'])
    y = df['target'].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    transform = FeatureTransformV1()
    transform.fit(X_train, y_train)

    base = baseline_pipeline()
    base.fit(X_train[['num_1', 'num_2']], y_train)

    predict_columns = transform.required_columns
    X_test_prod = X_test[predict_columns].copy()
    y_pred = transform.predict(X_test_prod)
    y_proba = transform.predict_proba(X_test_prod)[:, 1]
    acc = float(accuracy_score(y_test, y_pred))
    try:
        roc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        roc = float('nan')
    acc_base = float(accuracy_score(y_test, base.predict(X_test_prod[['num_1', 'num_2']])))

    return {
        'version': transform.version,
        'accuracy': acc,
        'roc_auc': roc,
        'accuracy_baseline': acc_base,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'example_input_rows': X_test_prod.iloc[:3].to_dict(orient='records'),
    }


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
