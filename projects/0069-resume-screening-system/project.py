import json
from typing import List, Dict, Any
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import precision_recall_fscore_support

SKILLS = ["python", "sql", "testing", "communication"]

def make_synthetic_resumes(n=400, seed=42):
    rng = np.random.default_rng(seed)
    templates = [
        "Experienced in {skills}. {extra}",
        "Worked with {skills} on projects. {extra}",
        "Hands-on: {skills}; interested in learning more. {extra}",
        "{extra} Skilled with {skills}.",
    ]
    extras = [
        "Led small teams.",
        "Open to remote roles.",
        "Contributed to tests and CI.",
        "Clear written and verbal communication.",
    ]
    # Controlled demographic-like cues (we include them only to test invariance; they are NOT used as features)
    cues = ["", "Available for relocation.", "Mother of two.", "Recently returned from sabbatical."]

    rows = []
    for i in range(n):
        templ = rng.choice(templates)
        extra = rng.choice(extras)
        cue = rng.choice(cues)
        # choose 0..len(SKILLS) skills to include, but bias towards having at least one
        count = rng.choice([1,1,1,2,0])
        chosen = rng.choice(SKILLS, size=count, replace=False).tolist() if count>0 else []
        # form a skills phrase with some variation
        if chosen:
            skills_phrase = ", ".join([s if rng.random() < 0.8 else s.upper() for s in chosen])
        else:
            skills_phrase = "general software practices"
        text = templ.format(skills=skills_phrase, extra=extra)
        if cue:
            text = cue + " " + text
        # ground truth: which skills mentioned literally (case-insensitive exact word match)
        truth = {skill: (skill in [w.strip().lower().strip('.,;:') for w in skills_phrase.split()]) for skill in SKILLS}
        rows.append({"id": i, "text": text, **truth})
    df = pd.DataFrame(rows)
    return df


def rule_based_evidence(text: str, skills: List[str]) -> Dict[str, List[str]]:
    # simple substring search for evidence spans; returns the matching substring(s) for each skill
    lower = text.lower()
    evidence = {}
    for skill in skills:
        hits = []
        if skill in lower:
            # return the exact slice(s) around the first occurrence
            idx = lower.find(skill)
            start = max(0, idx - 20)
            end = min(len(text), idx + len(skill) + 20)
            hits.append(text[start:end].strip())
        evidence[skill] = hits
    return evidence


def run(seed: int = 42) -> Dict[str, Any]:
    # 1) Prepare synthetic data
    df = make_synthetic_resumes(n=500, seed=seed)
    X = df["text"].values
    Y = df[SKILLS].astype(int).values

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=seed, stratify=Y.sum(axis=1))

    # 3) Text features and model: bag-of-words + One-vs-Rest logistic regression
    vect = CountVectorizer(ngram_range=(1,2), lowercase=True)
    Xtr = vect.fit_transform(X_train)
    Xte = vect.transform(X_test)

    clf = OneVsRestClassifier(LogisticRegression(max_iter=1000, random_state=seed))
    clf.fit(Xtr, y_train)

    y_pred = clf.predict(Xte)

    # 4) Evaluation: per-skill precision/recall/F1 and micro/macro aggregates
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, zero_division=0)
    metrics = {}
    for i, skill in enumerate(SKILLS):
        metrics[f"prec_{skill}"] = float(prec[i])
        metrics[f"rec_{skill}"] = float(rec[i])
        metrics[f"f1_{skill}"] = float(f1[i])
    micro_f1 = float(np.average(f1, weights=None))  # macro-like average (unweighted)
    macro_f1 = float(f1.mean())
    # also compute micro precision/recall/f1 for overall multi-label
    p_micro, r_micro, f_micro, _ = precision_recall_fscore_support(y_test.ravel(), y_pred.ravel(), average='binary', zero_division=0)
    # but above treats flattened labels — to provide a standard micro F1 we compute via sklearn's method
    from sklearn.metrics import f1_score, precision_score, recall_score
    try:
        micro_f1_score = float(f1_score(y_test, y_pred, average='micro', zero_division=0))
        micro_precision = float(precision_score(y_test, y_pred, average='micro', zero_division=0))
        micro_recall = float(recall_score(y_test, y_pred, average='micro', zero_division=0))
    except Exception:
        micro_f1_score = micro_precision = micro_recall = 0.0

    metrics.update({
        "micro_f1": micro_f1_score,
        "micro_precision": micro_precision,
        "micro_recall": micro_recall,
        "macro_f1": macro_f1,
        "num_test_samples": int(X_test.shape[0]),
    })

    # 5) Provide a small reviewable evidence preview from the rule-based extractor
    preview = []
    for i in range(min(8, X_test.shape[0])):
        txt = X_test[i]
        pred = {SKILLS[j]: bool(y_pred[i, j]) for j in range(len(SKILLS))}
        evidence = rule_based_evidence(txt, SKILLS)
        preview.append({"id": int(i), "text": txt, "predicted_present": pred, "evidence_spans": evidence})

    # 6) Return only JSON-serialisable metrics and a compact evidence preview for review
    out = {"metrics": metrics, "evidence_preview": preview}
    return out

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
