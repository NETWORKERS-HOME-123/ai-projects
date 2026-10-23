import json
from typing import Dict, List, Set
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def jaccard(a: Set[int], b: Set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)

def stability_from_sets(sets: List[Set[int]]) -> float:
    n = len(sets)
    if n < 2:
        return 1.0
    s = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            s += jaccard(sets[i], sets[j])
            count += 1
    return s / count if count else 0.0

def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)
    n_samples = 800
    n_informative = 5
    n_redundant = 3
    n_features = 15
    X, y = make_classification(n_samples=n_samples, n_features=n_features,
                               n_informative=n_informative, n_redundant=n_redundant,
                               n_repeated=0, n_classes=2, flip_y=0.05,
                               class_sep=1.0, random_state=seed)
    feature_names = [f"f{i}" for i in range(n_features)]
    groups = {}
    for i in range(n_features):
        if i < n_informative:
            groups[feature_names[i]] = "informative"
        elif i < n_informative + n_redundant:
            groups[feature_names[i]] = "redundant"
        else:
            groups[feature_names[i]] = "irrelevant"
    Xdf = pd.DataFrame(X, columns=feature_names)
    X_train, X_test, y_train, y_test = train_test_split(Xdf, y, test_size=0.25,
                                                        stratify=y, random_state=seed)
    k_select = 7
    select_pipeline = Pipeline([
        ("scale", StandardScaler()),
        ("filter", SelectKBest(score_func=f_classif, k=k_select)),
        ("clf", LogisticRegression(solver="liblinear", random_state=seed)),
    ])
    rfe_selector = RFE(estimator=LogisticRegression(solver="liblinear", random_state=seed),
                       n_features_to_select=7, step=1)
    rfe_pipeline = Pipeline([("scale", StandardScaler()), ("rfe", rfe_selector),
                             ("clf", LogisticRegression(solver="liblinear", random_state=seed))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    def cv_select_sets(pipeline, sel_name: str):
        fold_scores = []
        selected_sets = []
        for train_idx, val_idx in cv.split(X_train, y_train):
            Xt = X_train.iloc[train_idx]
            yt = y_train[train_idx]
            Xv = X_train.iloc[val_idx]
            yv = y_train[val_idx]
            pipeline.fit(Xt, yt)
            fold_scores.append(accuracy_score(yv, pipeline.predict(Xv)))
            if sel_name == "select":
                mask = pipeline.named_steps["filter"].get_support()
            else:
                mask = pipeline.named_steps["rfe"].get_support()
            selected_sets.append({i for i, m in enumerate(mask) if m})
        return float(np.mean(fold_scores)), fold_scores, selected_sets

    select_mean_cv, select_fold_scores, select_sets = cv_select_sets(select_pipeline, "select")
    rfe_mean_cv, rfe_fold_scores, rfe_sets = cv_select_sets(rfe_pipeline, "rfe")
    select_stability = stability_from_sets(select_sets)
    rfe_stability = stability_from_sets(rfe_sets)
    select_pipeline.fit(X_train, y_train)
    rfe_pipeline.fit(X_train, y_train)
    select_test_acc = float(accuracy_score(y_test, select_pipeline.predict(X_test)))
    rfe_test_acc = float(accuracy_score(y_test, rfe_pipeline.predict(X_test)))
    select_mask = select_pipeline.named_steps["filter"].get_support()
    rfe_mask = rfe_pipeline.named_steps["rfe"].get_support()
    select_features = [f for f, m in zip(feature_names, select_mask) if m]
    rfe_features = [f for f, m in zip(feature_names, rfe_mask) if m]
    baseline_pipe = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear", random_state=seed))])
    baseline_pipe.fit(X_train, y_train)
    baseline_acc = float(accuracy_score(y_test, baseline_pipe.predict(X_test)))
    group_names = ["informative", "redundant", "irrelevant"]
    group_effects = {}
    for g in group_names:
        cols_to_keep = [c for c in feature_names if groups[c] != g]
        p = Pipeline([("scale", StandardScaler()), ("clf", LogisticRegression(solver="liblinear", random_state=seed))])
        p.fit(X_train[cols_to_keep], y_train)
        acc = float(accuracy_score(y_test, p.predict(X_test[cols_to_keep])))
        group_effects[g] = {"accuracy_without_group": acc, "cost": baseline_acc - acc}
    audit = {"baseline_accuracy": baseline_acc, "selectk_retained": select_features,
             "rfe_retained": rfe_features, "group_effects": group_effects}
    results = {
        "selectk_mean_cv": float(select_mean_cv),
        "rfe_mean_cv": float(rfe_mean_cv),
        "selectk_test_accuracy": select_test_acc,
        "rfe_test_accuracy": rfe_test_acc,
        "selectk_feature_count": int(sum(select_mask)),
        "rfe_feature_count": int(sum(rfe_mask)),
        "selectk_stability": float(select_stability),
        "rfe_stability": float(rfe_stability),
        "audit": audit,
    }
    return results

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
