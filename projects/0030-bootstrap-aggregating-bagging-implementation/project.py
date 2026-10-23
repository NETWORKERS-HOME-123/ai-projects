import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score


def run(seed: int = 42) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    # 1. Prepare data (synthetic, reproducible)
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        flip_y=0.03,
        class_sep=1.0,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    # 2. Build experiment: vary number of estimators and collect scores
    n_list = [1, 5, 25, 100]
    holdout_scores = {}
    oob_scores = {}

    for n in n_list:
        base = DecisionTreeClassifier(random_state=seed)
        bag = BaggingClassifier(
            estimator=base,
            n_estimators=n,
            bootstrap=True,
            oob_score=True,
            random_state=seed,
            n_jobs=1,
        )
        bag.fit(X_train, y_train)
        # held-out accuracy
        y_pred = bag.predict(X_test)
        holdout_scores[f"holdout_accuracy_{n}"] = float(accuracy_score(y_test, y_pred))
        # oob_score_ is available since oob_score=True and bootstrap=True
        oob_scores[f"oob_score_{n}"] = float(getattr(bag, "oob_score_", float("nan")))

    # 3. Inspect bootstrap sample membership for the final model (n=100)
    final_n = 100
    final_bag = BaggingClassifier(
        estimator=DecisionTreeClassifier(random_state=seed),
        n_estimators=final_n,
        bootstrap=True,
        oob_score=True,
        random_state=seed,
        n_jobs=1,
    )
    final_bag.fit(X_train, y_train)

    # For each estimator, get the bootstrap sample indices used (private attribute)
    # sklearn stores samples of each estimator in _get_indices; we emulate membership via the rng
    # Instead, use the estimators_samples_ attribute provided by BaggingClassifier
    samples_mask = np.array(final_bag.estimators_samples_, dtype=bool)
    # fraction of training rows included per tree
    frac_included_per_tree = samples_mask.mean(axis=1)
    mean_frac_included = float(frac_included_per_tree.mean())

    # 4. Per-tree predictions on the held-out set and aggregate vote
    per_tree_preds = np.vstack([est.predict(X_test) for est in final_bag.estimators_])
    # majority vote (mode) across trees
    votes = per_tree_preds.mean(axis=0)  # mean in [0,1], >0.5 => class 1
    aggregated_pred = (votes > 0.5).astype(int)
    aggregated_accuracy = float(accuracy_score(y_test, aggregated_pred))
    # agreement: for each test row, fraction of trees that agree with aggregate
    agreement_per_row = np.mean(per_tree_preds == aggregated_pred, axis=0)
    mean_agreement = float(np.mean(agreement_per_row))

    # Package a compact summary of per-tree behaviour (small arrays summarised)
    summary = {
        "holdout_scores": holdout_scores,
        "oob_scores": {k: oob_scores.get(k, float('nan')) for k in oob_scores},
        "final_n": final_n,
        "final_holdout_accuracy": holdout_scores[f"holdout_accuracy_{final_n}"],
        "final_oob_score": oob_scores.get(f"oob_score_{final_n}", float('nan')),
        "mean_fraction_included_per_tree": mean_frac_included,
        "aggregated_holdout_accuracy": aggregated_accuracy,
        "mean_tree_agreement_with_aggregate": mean_agreement,
        # small deterministic sample of per-tree predictions for the first 10 test rows
        "per_tree_preds_first_10_rows": per_tree_preds[:, :10].astype(int).tolist(),
    }

    # Return a flat dict for tests checking specific keys
    result = {
        "holdout_accuracy_100": float(summary["final_holdout_accuracy"]),
        "oob_score_100": float(summary["final_oob_score"]),
        "mean_fraction_included_per_tree": float(summary["mean_fraction_included_per_tree"]),
        "aggregated_holdout_accuracy": float(summary["aggregated_holdout_accuracy"]),
        "mean_tree_agreement_with_aggregate": float(summary["mean_tree_agreement_with_aggregate"]),
        "per_tree_preds_first_10_rows": summary["per_tree_preds_first_10_rows"],
        "all": summary,
    }

    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
