import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report


def _build_synthetic_corpus(random_state: int = 42):
    rng = np.random.RandomState(random_state)
    # Define prototypes per emotion with simple stylistic variants
    prototypes = {
        "joy": [
            "I am so happy today!",
            "This made me smile.",
            "Feeling great and delighted",
            "Yay! That was awesome :)",
            "What a wonderful surprise"
        ],
        "sadness": [
            "I feel really down.",
            "This is so sad.",
            "I'm disappointed and low",
            "Tears came to my eyes",
            "I miss those days"
        ],
        "anger": [
            "This makes me angry!",
            "I'm furious right now.",
            "How dare they?",
            "So annoyed and pissed off",
            "That was unacceptable"
        ],
        "fear": [
            "I'm scared about the results",
            "This makes me nervous.",
            "I'm worried and anxious",
            "I fear what might happen",
            "My heart raced with panic"
        ],
        "neutral": [
            "I will attend the meeting.",
            "The package arrived today.",
            "It is scheduled for Monday.",
            "I took notes during class.",
            "The temperature is moderate"
        ],
        # Ambiguous examples mix cues intentionally
        "ambiguous": [
            "I can't tell if I'm happy or relieved.",
            "Well, that was something...",
            "Not sure how I feel about this",
            "Mixed feelings about the result",
            "It was okay, I guess"
        ]
    }

    texts = []
    labels = []

    # For each prototype, create multiple variants (add punctuation, slang, short forms)
    for label, seeds in prototypes.items():
        for s in seeds:
            for i in range(6):  # create variants -> enough data for a small train/test split
                text = s
                # randomly add stylistic noise
                if rng.rand() < 0.4:
                    text = text + rng.choice([" :)", " :(", "!!", "..."])
                if rng.rand() < 0.3:
                    text = text.replace("I am", "I'm")
                if rng.rand() < 0.2:
                    text = text.lower()
                # small chance to inject slang
                if rng.rand() < 0.15:
                    text = text.replace("happy", "stoked").replace("sad", "meh").replace("angry", "annoyed")
                texts.append(text)
                labels.append(label)

    # Shuffle and return
    df = pd.DataFrame({"text": texts, "label": labels})
    df = df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return df


def run(seed: int = 42) -> Dict:
    """Train a small text classifier on a synthetic emotion corpus and evaluate.

    Returns a JSON-serialisable dict with per-label F1, macro F1 (main metric),
    a short classification report string, and a list of ambiguous examples kept for review.
    """
    rng = np.random.RandomState(seed)

    df = _build_synthetic_corpus(random_state=seed)

    X = df["text"].values
    y = df["label"].values

    # Hold-out split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    # Vectorise text
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X_train_vec = vec.fit_transform(X_train)
    X_test_vec = vec.transform(X_test)

    # Classifier with a fixed random state for determinism
    # Some sklearn versions may not accept multi_class in the signature used originally;
    # use solver explicitly which is widely supported.
    clf = LogisticRegression(max_iter=1000, random_state=seed, C=1.0, solver="lbfgs")
    clf.fit(X_train_vec, y_train)

    y_pred = clf.predict(X_test_vec)

    labels = sorted(list(set(y)))
    per_label_f1_vals = f1_score(y_test, y_pred, average=None, labels=labels)
    per_label_f1 = {lab: float(f) for lab, f in zip(labels, per_label_f1_vals)}
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))

    # Create a short text report (kept small and JSON-serialisable)
    report = classification_report(y_test, y_pred, labels=labels, zero_division=0)

    # Preserve ambiguous examples from the test set for review (those labelled 'ambiguous')
    ambiguous_mask = (y_test == "ambiguous")
    ambiguous_examples = list(X_test[ambiguous_mask])

    results = {
        "macro_f1": macro_f1,
        "per_label_f1": per_label_f1,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "classification_report": report,
        "ambiguous_examples_for_review": ambiguous_examples,
    }

    return results


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
