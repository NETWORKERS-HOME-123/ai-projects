import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix


def _make_synthetic_messages(n_samples: int = 400, seed: int = 42):
    rng = np.random.RandomState(seed)
    spam_templates = [
        "win a free prize now",
        "limited offer claim your reward",
        "urgent account verify now",
        "congratulations you have won",
        "free trial click link",
        "cheap meds no prescription",
        "earn money from home",
        "exclusive deal just for you",
    ]
    ham_templates = [
        "are we meeting tomorrow",
        "let's have lunch",
        "can you review the document",
        "happy birthday see you soon",
        "I'll call you later",
        "thanks for the update",
        "please send the report",
        "that sounds like a plan",
    ]
    texts = []
    labels = []
    for _ in range(n_samples):
        if rng.rand() < 0.4:  # make spam somewhat less frequent but present
            t = rng.choice(spam_templates)
            # insert some variability
            extras = ["now", "today", "click", "link", "offer", "win", "free"]
            if rng.rand() < 0.5:
                t = t + " " + rng.choice(extras)
            texts.append(t)
            labels.append(1)
        else:
            t = rng.choice(ham_templates)
            if rng.rand() < 0.3:
                t = t + " " + rng.choice(["ok", "thanks", "soon", "tomorrow"])
            texts.append(t)
            labels.append(0)
    return pd.DataFrame({"text": texts, "label": labels})


def run(seed: int = 42) -> Dict:
    # Build data
    df = _make_synthetic_messages(n_samples=400, seed=seed)

    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Vectorise text: fit only on training data
    vect = CountVectorizer()
    X_train_counts = vect.fit_transform(X_train)
    X_test_counts = vect.transform(X_test)

    # Inspect top tokens by vocabulary order frequency on training data
    token_counts = np.asarray(X_train_counts.sum(axis=0)).ravel()
    vocab = np.array([t for t, _ in sorted(vect.vocabulary_.items(), key=lambda kv: kv[1])])
    # compute mapping index->token
    idx_to_token = {i: token for token, i in vect.vocabulary_.items()}
    top_indices = token_counts.argsort()[::-1][:10]
    top_tokens: List[str] = [idx_to_token[int(i)] for i in top_indices]

    # Train MultinomialNB
    clf = MultinomialNB(alpha=1.0)
    clf.fit(X_train_counts, y_train)

    # Majority class baseline accuracy
    majority_pred = np.full_like(y_test, fill_value=np.bincount(y_train).argmax())
    majority_accuracy = float(accuracy_score(y_test, majority_pred))

    # Evaluate on test set
    y_pred = clf.predict(X_test_counts)
    accuracy = float(accuracy_score(y_test, y_pred))
    # spam is label 1: recall for spam
    spam_recall = float(recall_score(y_test, y_pred, pos_label=1))
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    # false positive rate for ham = FP / (FP + TN)
    ham_false_positive_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "accuracy": accuracy,
        "spam_recall": spam_recall,
        "ham_false_positive_rate": ham_false_positive_rate,
        "majority_accuracy": float(majority_accuracy),
        "top_tokens": top_tokens,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
