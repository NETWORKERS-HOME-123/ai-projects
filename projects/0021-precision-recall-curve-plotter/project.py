import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score, precision_score, auc


def run(seed: int = 42) -> Dict:
    rng = np.random.RandomState(seed)

    # 1) Prepare the data: low-prevalence synthetic binary classification
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=4,
        n_redundant=2,
        n_repeated=0,
        n_classes=2,
        weights=[0.95, 0.05],  # low positive-class prevalence (~5%)
        class_sep=1.0,
        flip_y=0.01,
        random_state=seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=seed
    )

    # 2) Build the experiment: fit a probabilistic classifier
    clf = LogisticRegression(max_iter=1000, random_state=seed)
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]

    # Compute precision-recall curve and average precision
    precision, recall, thresholds = precision_recall_curve(y_test, scores)
    avg_prec = float(average_precision_score(y_test, scores))

    # Area under PR curve by trapezoidal rule (precision as function of recall)
    # Use sklearn.metrics.auc for robust availability across numpy versions
    # precision_recall_curve returns recall in descending order; auc will handle ordering
    pr_auc_trap = float(auc(recall, precision))

    prevalence = float(np.mean(y_test))

    # 3) Evaluate result: examine low-prevalence scenarios by selecting a few score thresholds
    # Choose thresholds at specific score quantiles so thresholds adapt to score distribution
    quantiles = [0.5, 0.75, 0.9]
    chosen_thresholds = [float(np.quantile(scores, q)) for q in quantiles]

    thresholds_info: List[Dict] = []
    for t in chosen_thresholds:
        preds = scores >= t
        pred_count = int(preds.sum())
        prec_at_t = float(precision_score(y_test, preds)) if pred_count > 0 else 0.0
        recall_at_t = float((y_test[preds].sum() / max(1, int(y_test.sum())))) if pred_count > 0 else 0.0
        thresholds_info.append({
            "threshold": t,
            "predicted_positives": pred_count,
            "precision": prec_at_t,
            "recall_of_positives": recall_at_t,
        })

    # 4) Define reviewable output: given candidate review-queue sizes, estimate expected errors
    # We simulate a reviewer who looks at top-k scored items (highest scores). For each k we compute
    # how many false positives are expected in that queue.
    queue_sizes = [10, 50, 100]
    order = np.argsort(-scores)
    review_queue: List[Dict] = []
    for k in queue_sizes:
        k_use = min(k, len(scores))
        topk_idx = order[:k_use]
        true_positives_in_topk = int(y_test[topk_idx].sum())
        false_positives_in_topk = int(k_use - true_positives_in_topk)
        # expected error rate in the review queue
        error_rate = float(false_positives_in_topk / max(1, k_use))
        review_queue.append({
            "queue_size": k_use,
            "true_positives": true_positives_in_topk,
            "false_positives": false_positives_in_topk,
            "error_rate": error_rate,
        })

    result = {
        "average_precision": avg_prec,
        "pr_auc_trap": pr_auc_trap,
        "prevalence": prevalence,
        "precision_recall_points": {
            "precision_length": int(len(precision)),
            "recall_length": int(len(recall)),
            "thresholds_length": int(len(thresholds)),
        },
        "thresholds_info": thresholds_info,
        "review_queue": review_queue,
        "seed": int(seed),
    }

    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
