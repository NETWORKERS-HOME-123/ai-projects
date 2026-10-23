import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, recall_score


def _make_synthetic_threads(n_threads: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    folders = ["Work", "Personal", "Promotions", "Spam"]
    topic_words = {
        "Work": ["meeting", "project", "deadline", "client", "report", "schedule"],
        "Personal": ["dinner", "party", "family", "photo", "weekend", "congrats"],
        "Promotions": ["sale", "offer", "discount", "deal", "coupon", "exclusive"],
        "Spam": ["winner", "lottery", "prize", "urgent", "claim", "free"]
    }
    templates = [
        "{lead} {mid} {tail}.",
        "{lead} - please see {tail}.",
        "{lead}: {mid} about {tail}.",
        "Quick note {lead} {tail}"
    ]
    rows = []
    for tid in range(n_threads):
        # assign a dominant folder for this thread
        folder = rng.choice(folders, p=[0.35, 0.25, 0.25, 0.15])
        # thread length between 1 and 4
        tlen = rng.randint(1, 5)
        for msg_idx in range(tlen):
            lead = rng.choice(topic_words[folder])
            mid = rng.choice(topic_words[folder])
            # inject occasional cross-topic words to simulate noise
            if rng.rand() < 0.2:
                other = rng.choice([f for f in folders if f != folder])
                tail = rng.choice(topic_words[other])
            else:
                tail = rng.choice(topic_words[folder])
            template = rng.choice(templates)
            text = template.format(lead=lead, mid=mid, tail=tail)
            # add small filler
            if rng.rand() < 0.3:
                text += " " + rng.choice(["Please respond", "See attached", "FYI", "Thanks"]) 
            rows.append({"thread_id": f"t{tid}", "text": text, "folder": folder})
    return pd.DataFrame(rows)


def run(seed: int = 42) -> Dict:
    """Run the experiment. Returns a JSON-serialisable dict of metrics and sample suggestions.

    The function is deterministic for a fixed seed.
    """
    rng = np.random.RandomState(seed)
    # 1. Prepare data: synthetic threads
    df = _make_synthetic_threads(n_threads=240, seed=seed)
    # shuffle thread ids and split whole threads to keep conversation together
    thread_ids = df["thread_id"].unique()
    rng.shuffle(thread_ids)
    n_train_threads = int(len(thread_ids) * 0.7)
    train_threads = set(thread_ids[:n_train_threads])
    df["split"] = df["thread_id"].apply(lambda t: "train" if t in train_threads else "test")

    X_train = df.loc[df["split"] == "train", "text"].tolist()
    y_train = df.loc[df["split"] == "train", "folder"].tolist()
    X_test = df.loc[df["split"] == "test", "text"].tolist()
    y_test = df.loc[df["split"] == "test", "folder"].tolist()

    # 2. Build experiment: TF-IDF + Logistic Regression
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        ("clf", LogisticRegression(max_iter=200, random_state=seed))
    ])
    pipe.fit(X_train, y_train)

    # Predict probabilities, then apply an uncertainty threshold
    prob = pipe.predict_proba(X_test)
    classes = list(pipe.named_steps["clf"].classes_)
    top_idx = np.argmax(prob, axis=1)
    top_prob = prob[np.arange(len(prob)), top_idx]
    pred = [classes[i] for i in top_idx]
    # label as uncertain if top_prob < threshold
    threshold = 0.60
    suggestion_labels: List[str] = []
    suggestion_confidence: List[float] = []
    suggestion_uncertain: List[bool] = []
    for p_label, p_conf in zip(pred, top_prob):
        suggestion_labels.append(p_label)
        suggestion_confidence.append(float(p_conf))
        suggestion_uncertain.append(bool(p_conf < threshold))

    # 3. Evaluate: accuracy and per-folder recall
    acc = float(accuracy_score(y_test, suggestion_labels))
    recalls = recall_score(y_test, suggestion_labels, average=None, labels=classes)
    per_folder_recall = {c: float(r) for c, r in zip(classes, recalls)}
    macro_recall = float(np.mean(recalls))

    # Identify some important misclassifications: high-confidence wrong predictions
    mis_idx = [i for i, (gt, predc, conf) in enumerate(zip(y_test, suggestion_labels, suggestion_confidence)) if gt != predc and conf > 0.7]
    sample_mis = []
    for i in mis_idx[:5]:
        sample_mis.append({"text": X_test[i], "true": y_test[i], "pred": suggestion_labels[i], "conf": suggestion_confidence[i]})

    # 4. Reviewable output: suggestions include label, confidence and uncertain flag
    suggestions = []
    for t, lab, conf, unc in zip(df.loc[df["split"] == "test", "thread_id"].tolist(), suggestion_labels, suggestion_confidence, suggestion_uncertain):
        suggestions.append({"thread_id": t, "suggested_label": lab, "confidence": conf, "uncertain": unc})

    result = {
        "accuracy": acc,
        "macro_recall": macro_recall,
        "per_folder_recall": per_folder_recall,
        "n_test_messages": int(len(X_test)),
        "n_train_messages": int(len(X_train)),
        "sample_high_conf_misclassifications": sample_mis,
        "some_suggestions": suggestions[:6]
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
