import json
import random
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score


def _make_synthetic_corpus(seed: int = 42) -> pd.DataFrame:
    """Create a small synthetic news-like corpus with publishers and timestamps.

    This is intentionally small and synthetic for teaching purposes.
    """
    rng = random.Random(seed)
    np.random.seed(seed)

    topics = {
        "politics": [
            "government passes new bill on climate policy",
            "senators debate budget and public spending",
            "election campaigns focus on healthcare and taxes",
        ],
        "sports": [
            "local team wins championship after dramatic final",
            "star player signs new contract with club",
            "coach praises young talent after the victory",
        ],
        "technology": [
            "startup announces a new open source toolkit",
            "major firm unveils next generation smartphone",
            "researchers publish new results on machine learning",
        ],
        "business": [
            "company reports quarterly earnings above forecasts",
            "merger talks continue between two large firms",
            "market analysts revise outlook for industry growth",
        ],
    }

    publishers = ["NewsA", "NewsB", "SyndicateX"]

    rows: List[Dict] = []
    # create roughly 60 samples by repeating templates with small variations
    for i in range(60):
        topic = rng.choice(list(topics.keys()))
        template = rng.choice(topics[topic])
        # add a short modifier to create variety
        modifier = rng.choice(["", " Analysts say", " in local area", "; experts comment"])
        text = template + modifier
        # simulate syndicated duplication: SyndicateX distributes some identical items
        if rng.random() < 0.12:
            publisher = "SyndicateX"
        else:
            publisher = rng.choice([p for p in publishers if p != "SyndicateX"])
        # timestamp: integer day; distribute over 30 days
        day = int(np.clip(np.random.normal(15, 7), 1, 30))
        rows.append({"id": i, "text": text, "topic": topic, "publisher": publisher, "day": day})

    df = pd.DataFrame(rows)

    # Create some exact duplicates (syndicated) to test deduplication
    # Pick a few rows to duplicate across publishers
    dup_samples = df.sample(6, random_state=seed)
    next_id = int(df["id"].max() + 1)
    for _, r in dup_samples.iterrows():
        for pub in ["NewsA", "NewsB", "SyndicateX"]:
            df = pd.concat([df, pd.DataFrame([{"id": next_id, "text": r.text, "topic": r.topic, "publisher": pub, "day": r.day}])], ignore_index=True)
            next_id += 1

    # ensure deterministic ordering
    df = df.sort_values(["day", "publisher", "id"]).reset_index(drop=True)
    return df


def run(seed: int = 42) -> Dict:
    """Train a TF-IDF + Logistic Regression multiclass classifier on synthetic news snippets.

    Returns a JSON-serialisable dict with metrics and an example of editable suggestions.
    """
    df = _make_synthetic_corpus(seed=seed)

    # Deduplicate exact syndicated copies by (text, day)
    df = df.drop_duplicates(subset=["text", "day"]).reset_index(drop=True)

    # Temporal split: early days -> train, later days -> test
    cutoff_day = df["day"].quantile(0.7)
    train_df = df[df["day"] <= cutoff_day].copy()
    test_df = df[df["day"] > cutoff_day].copy()

    # Held-out publisher evaluation: hold out 'SyndicateX' as an example publisher
    holdout_publisher = "SyndicateX"
    # Create a train set that excludes the holdout publisher rows (for the publisher test)
    train_no_hold = df[df["publisher"] != holdout_publisher].copy()
    test_hold = df[df["publisher"] == holdout_publisher].copy()

    # Use separate vectorizers for the temporal split and the publisher-held-out split
    vectoriser_time = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=2000)
    vectoriser_pub = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_features=2000)

    # some sklearn builds used in teaching envs may not accept the multi_class argument; keep defaults for compatibility
    clf = LogisticRegression(max_iter=1000, random_state=seed)

    # Fit on temporal train split
    X_train = vectoriser_time.fit_transform(train_df["text"])
    clf.fit(X_train, train_df["topic"])

    X_test = vectoriser_time.transform(test_df["text"]) if not test_df.empty else vectoriser_time.transform(train_df["text"])
    y_pred = clf.predict(X_test)

    macro_f1_time = float(f1_score(test_df["topic"], y_pred, average="macro")) if not test_df.empty else 0.0
    acc_time = float(accuracy_score(test_df["topic"], y_pred)) if not test_df.empty else 0.0

    # Held-out publisher evaluation: train on all non-holdout, test on holdout
    X_tr2 = vectoriser_pub.fit_transform(train_no_hold["text"]) if not train_no_hold.empty else vectoriser_pub.fit_transform(train_df["text"])
    clf2 = LogisticRegression(max_iter=1000, random_state=seed)
    if not train_no_hold.empty:
        clf2.fit(X_tr2, train_no_hold["topic"]) 
    else:
        clf2.fit(X_tr2, train_df["topic"])

    X_hold = vectoriser_pub.transform(test_hold["text"]) if not test_hold.empty else vectoriser_pub.transform(train_no_hold["text"])
    y_hold_pred = clf2.predict(X_hold)

    macro_f1_publisher = float(f1_score(test_hold["topic"], y_hold_pred, average="macro")) if not test_hold.empty else 0.0
    acc_publisher = float(accuracy_score(test_hold["topic"], y_hold_pred)) if not test_hold.empty else 0.0

    # Inspect top tokens per class (simple approach)
    feature_names = np.array(vectoriser_pub.get_feature_names_out())
    coefs = clf2.coef_ if hasattr(clf2, "coef_") else np.zeros((len(np.unique(df["topic"])), feature_names.size))
    class_labels = clf2.classes_.tolist()
    top_tokens = {}
    for i, label in enumerate(class_labels):
        top_idx = np.argsort(coefs[i])[-6:][::-1]
        top_tokens[label] = feature_names[top_idx].tolist()

    # Produce editable topic suggestions for three example items from the temporal test set
    sample_suggestions = []
    sample_rows = test_df.sample(n=min(3, len(test_df)), random_state=seed) if not test_df.empty else train_df.sample(n=3, random_state=seed)
    probs = clf.predict_proba(vectoriser_time.transform(sample_rows["text"]))
    for idx, (_, row) in enumerate(sample_rows.iterrows()):
        prob = {lab: float(p) for lab, p in zip(clf.classes_, probs[idx])}
        # suggestions are ordered labels with probabilities; remind they are suggestions
        ordered = sorted(prob.items(), key=lambda x: -x[1])
        sample_suggestions.append({"id": int(row["id"]), "text": row["text"], "suggestions": ordered})

    result = {
        "macro_f1_time": round(macro_f1_time, 4),
        "accuracy_time": round(acc_time, 4),
        "macro_f1_publisher": round(macro_f1_publisher, 4),
        "accuracy_publisher": round(acc_publisher, 4),
        "top_tokens_per_class": top_tokens,
        "sample_suggestions": sample_suggestions,
        "note": "Suggestions are editable labels for review; probabilities are not credibility scores.",
    }
    return result


if __name__ == "__main__":
    import sys

    out = run(42)
    print(json.dumps(out, indent=2))
