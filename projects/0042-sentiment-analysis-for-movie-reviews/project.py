import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score


def _make_synthetic_reviews(seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    pos_templates = [
        "I loved the film, it was brilliant and moving.",
        "A delightful experience with strong acting and warm humour.",
        "Exceptional visuals and a touching story — highly recommend.",
        "An entertaining film with charming performances.",
        "Great soundtrack and memorable scenes.",
    ]
    neg_templates = [
        "I hated the movie; it was boring and long.",
        "Poor writing and flat characters ruined it for me.",
        "A tedious film with little to enjoy.",
        "Bad pacing and a forgettable score.",
        "Disappointing: wasted potential and weak direction.",
    ]
    # Negation constructions
    negation = [
        "not good",
        "not convincing",
        "not worth the time",
    ]
    # Sarcasm-like short templates
    sarcasm = [
        "Loved the plot — said no one ever.",
        "An absolute masterpiece (if you like boredom).",
    ]
    # Mixed opinion templates
    mixed = [
        "Great visuals but the plot was weak.",
        "Strong acting; weak screenplay.",
        "Lovely score, terrible pacing.",
    ]

    rows = []
    # Create many variants by slight random swaps and punctuation/noise
    for i in range(25):
        p = rng.choice(pos_templates)
        rows.append((p, 1))
        n = rng.choice(neg_templates)
        rows.append((n, 0))
    # Add targeted phenomena
    for t in negation:
        rows.append((f"The movie was {t}.", 0))
    for s in sarcasm:
        rows.append((s, 0))
    for m in mixed:
        rows.append((m, 0))
        rows.append((m.replace('weak', 'great'), 1))
    # Introduce a few near-duplicates intentionally
    rows.append(("A delightful experience with strong acting and warm humour.", 1))
    rows.append(("A delightful experience with strong acting and warm humour!", 1))

    df = pd.DataFrame(rows, columns=["text", "label"])
    # Simple cleaning: strip and normalize whitespace
    df["text"] = df["text"].str.strip()
    # Remove exact duplicates (as per requirement)
    df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
    return df


def run(seed: int = 42) -> Dict:
    """Train a TF-IDF + Logistic Regression baseline on synthetic movie reviews.

    Returns a JSON-serialisable dict with metrics and a small review panel.
    """
    df = _make_synthetic_reviews(seed)
    X = df["text"].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y if len(np.unique(y)) > 1 else None
    )

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    X_train_t = vec.fit_transform(X_train)
    X_test_t = vec.transform(X_test)

    clf = LogisticRegression(random_state=seed, max_iter=1000)
    clf.fit(X_train_t, y_train)

    y_pred = clf.predict(X_test_t)
    y_proba = clf.predict_proba(X_test_t)[:, 1]

    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    acc = float(accuracy_score(y_test, y_pred))

    # Inspect specific phenomena in the held-out set
    def examples_of(kind: str) -> List[int]:
        idxs = []
        for i, t in enumerate(X_test):
            s = t.lower()
            if kind == "negation" and "not " in s:
                idxs.append(i)
            if kind == "sarcasm" and ("said no one" in s or "masterpiece (if" in s):
                idxs.append(i)
            if kind == "mixed" and ("but" in s or ";" in s):
                idxs.append(i)
        return idxs

    panel = []
    for kind in ("negation", "sarcasm", "mixed"):
        ids = examples_of(kind)
        items = []
        for i in ids[:3]:
            items.append(
                {
                    "text": X_test[i],
                    "label": int(y_test[i]),
                    "pred": int(y_pred[i]),
                    "prob_pos": float(round(y_proba[i], 4)),
                }
            )
        panel.append({"kind": kind, "examples": items, "found": len(ids)})

    # Review-summary panel: a few random held-out examples for manual correction
    rng = np.random.RandomState(seed)
    sample_ids = rng.choice(len(X_test), size=min(5, len(X_test)), replace=False)
    review_panel = []
    for i in sample_ids:
        review_panel.append(
            {
                "text": X_test[i],
                "label": int(y_test[i]),
                "pred": int(y_pred[i]),
                "prob_pos": float(round(y_proba[i], 4)),
            }
        )

    result = {
        "num_examples": int(len(df)),
        "num_train": int(X_train.shape[0]),
        "num_test": int(X_test.shape[0]),
        "accuracy": acc,
        "macro_f1": macro_f1,
        "panel_inspection": panel,
        "review_panel": review_panel,
    }
    return result


if __name__ == "__main__":
    import sys

    out = run()
    print(json.dumps(out, indent=2))
