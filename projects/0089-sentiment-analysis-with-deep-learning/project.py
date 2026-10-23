import json
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, brier_score_loss
from sklearn.neural_network import MLPClassifier


def _build_synthetic_sentences(seed: int = 42) -> pd.DataFrame:
    # A small synthetic corpus with explicit negation and sentiment words.
    rng = np.random.RandomState(seed)
    positive_adjs = ["good", "great", "excellent", "pleasant", "wonderful"]
    negative_adjs = ["bad", "terrible", "awful", "poor", "horrible"]
    topics = ["service", "food", "product", "delivery", "experience"]
    templates = [
        "The {topic} was {adj}.",
        "I had a {adj} {topic}.",
        "Not a {adj} {topic}.",
        "This {topic} is {adj}.",
        "I would recommend this {topic}, it was {adj}.",
    ]

    rows = []
    # Create balanced examples, include negation patterns
    for _ in range(120):
        topic = rng.choice(topics)
        if rng.rand() < 0.5:
            adj = rng.choice(positive_adjs)
            label = 1
        else:
            adj = rng.choice(negative_adjs)
            label = 0
        tmpl = rng.choice(templates)
        text = tmpl.format(topic=topic, adj=adj)
        # occasionally add explicit negation to flip polarity
        if rng.rand() < 0.12:
            # e.g. "Not a good service" becomes negative
            text = "Not " + text
            label = 0 if label == 1 else 1
        # occasional intensifier
        if rng.rand() < 0.08:
            text = "Very " + text
        rows.append((text, label))

    df = pd.DataFrame(rows, columns=["text", "label"])    
    # shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df


def _train_and_eval(seed: int = 42) -> Dict[str, object]:
    df = _build_synthetic_sentences(seed)

    # Time-aware split is unnecessary for synthetic data; we hold out 20% for test.
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # Build TF-IDF using training-only vocabulary
    tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Baseline: logistic regression
    baseline = LogisticRegression(random_state=seed, max_iter=1000)
    t0 = time.perf_counter()
    baseline.fit(X_train_tfidf, y_train)
    t1 = time.perf_counter()
    baseline_train_time = t1 - t0

    t0 = time.perf_counter()
    proba_baseline = baseline.predict_proba(X_test_tfidf)[:, 1]
    t1 = time.perf_counter()
    baseline_infer_time = t1 - t0
    pred_baseline = (proba_baseline >= 0.5).astype(int)

    # Neural model: small MLP on TF-IDF features
    mlp = MLPClassifier(hidden_layer_sizes=(32,), max_iter=200, random_state=seed)
    t0 = time.perf_counter()
    mlp.fit(X_train_tfidf, y_train)
    t1 = time.perf_counter()
    mlp_train_time = t1 - t0

    t0 = time.perf_counter()
    proba_mlp = mlp.predict_proba(X_test_tfidf)[:, 1]
    t1 = time.perf_counter()
    mlp_infer_time = t1 - t0
    pred_mlp = (proba_mlp >= 0.5).astype(int)

    # Metrics
    f1_mlp = float(f1_score(y_test, pred_mlp, average="macro"))
    f1_baseline = float(f1_score(y_test, pred_baseline, average="macro"))

    # Brier score as simple calibration proxy
    brier_mlp = float(brier_score_loss(y_test, proba_mlp))
    brier_baseline = float(brier_score_loss(y_test, proba_baseline))

    # Inference cost ratio (per-sample average)
    per_sample_mlp = mlp_infer_time / max(1, X_test_tfidf.shape[0])
    per_sample_baseline = baseline_infer_time / max(1, X_test_tfidf.shape[0])
    infer_time_ratio = float(per_sample_mlp / (per_sample_baseline + 1e-12))

    # Prepare small qualitative review output (first 5 test samples)
    review_samples: List[Dict[str, object]] = []
    for text, yt, pb, pm, pb_pred, pm_pred in zip(
        X_test[:5], y_test[:5], proba_baseline[:5], proba_mlp[:5], pred_baseline[:5], pred_mlp[:5]
    ):
        review_samples.append(
            {
                "text": str(text),
                "label": int(yt),
                "baseline_proba": float(pb),
                "baseline_pred": int(pb_pred),
                "mlp_proba": float(pm),
                "mlp_pred": int(pm_pred),
            }
        )

    results = {
        "macro_f1_neural": f1_mlp,
        "macro_f1_tfidf": f1_baseline,
        "brier_neural": brier_mlp,
        "brier_tfidf": brier_baseline,
        "infer_time_ratio_mlp_over_baseline": infer_time_ratio,
        "train_time_neural": float(mlp_train_time),
        "train_time_tfidf": float(baseline_train_time),
        "review_samples": review_samples,
        "n_test": int(X_test.shape[0]),
    }
    return results


def run(seed: int = 42) -> Dict[str, object]:
    return _train_and_eval(seed)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
