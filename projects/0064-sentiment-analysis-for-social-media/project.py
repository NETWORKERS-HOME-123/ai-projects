import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def build_synthetic_dataset(seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    # Minimal anonymised, synthetic social-media-like posts with slang and code-mixing
    rows = [
        ("luv this! so good 😊", "en", "early", "positive"),
        ("totally lame, ugh", "en", "early", "negative"),
        ("meh idk, bit mixed vibes", "en", "early", "mixed"),
        ("great job yaar! so proud", "hi-en", "late", "positive"),
        ("nah bro, that was bad af", "en", "late", "negative"),
        ("kinda okay, could be better", "en", "late", "mixed"),
        ("awesome pic! love it", "en", "early", "positive"),
        ("not good, disappointed", "en", "late", "negative"),
        ("thoda confusing lol", "hi-en", "early", "mixed"),
        ("best day ever!!!", "en", "late", "positive"),
        ("sucks man, so sad", "en", "early", "negative"),
        ("idk what to feel 🤷", "en", "late", "mixed"),
    ]
    # Duplicate with small perturbations to make learnable patterns
    df = pd.DataFrame(rows, columns=["text", "language", "period", "label"])
    # Expand dataset deterministically
    extras = []
    for i in range(30):
        idx = rng.randint(len(df))
        text = df.loc[idx, "text"]
        # slight, deterministic noising
        if rng.rand() < 0.3:
            text = text + " :)"
        if rng.rand() < 0.2:
            text = text.replace(" ", " ")
        extras.append((text, df.loc[idx, "language"], df.loc[idx, "period"], df.loc[idx, "label"]))
    df = pd.concat([df, pd.DataFrame(extras, columns=df.columns)], ignore_index=True)
    return df


def run(seed: int = 42) -> Dict[str, Any]:
    # Build data
    df = build_synthetic_dataset(seed)
    X = df["text"].values
    y = df["label"].values
    langs = df["language"].values
    periods = df["period"].values

    # Split
    X_train, X_test, y_train, y_test, lang_train, lang_test, per_train, per_test = train_test_split(
        X, y, langs, periods, test_size=0.25, random_state=seed, stratify=y
    )

    # Vectorise
    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2000)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # Classifier
    # some sklearn versions do not accept explicit multi_class keyword in constructor; rely on defaults
    clf = LogisticRegression(solver='lbfgs', max_iter=1000, random_state=seed)
    clf.fit(Xtr, y_train)

    # Predictions and metrics
    probs = clf.predict_proba(Xte)
    preds = clf.classes_[np.argmax(probs, axis=1)]
    overall_macro_f1 = float(f1_score(y_test, preds, average='macro'))

    # Per-language and per-period macro F1
    per_lang = {}
    for l in np.unique(lang_test):
        mask = lang_test == l
        if mask.sum() == 0:
            per_lang[l] = None
        else:
            per_lang[l] = float(f1_score(y_test[mask], preds[mask], average='macro'))

    per_period = {}
    for p in np.unique(per_test):
        mask = per_test == p
        if mask.sum() == 0:
            per_period[p] = None
        else:
            per_period[p] = float(f1_score(y_test[mask], preds[mask], average='macro'))

    # Representative anonymised examples: pick test rows with low max probability (uncertain)
    max_probs = probs.max(axis=1)
    order = np.argsort(max_probs)
    examples = []
    for i in order[:5]:
        examples.append({
            "text": str(X_test[i]),
            "true_label": str(y_test[i]),
            "predicted_label": str(preds[i]),
            "predicted_prob": float(max_probs[i])
        })

    result = {
        "overall_macro_f1": overall_macro_f1,
        "per_language_macro_f1": per_lang,
        "per_period_macro_f1": per_period,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "examples_uncertain": examples
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
