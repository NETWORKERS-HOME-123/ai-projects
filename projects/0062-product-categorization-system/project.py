import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix


def _make_synthetic_catalog(seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    rng = np.random.RandomState(seed)
    taxonomy = {
        "Electronics": ["smartphone", "laptop", "tablet"],
        "Home": ["sofa", "chair", "table"],
        "Apparel": ["tshirt", "jeans", "jacket"]
    }
    templates = {
        "smartphone": [
            "{adj} smartphone with {feat}", "{brand} {adj} phone, {feat}", "{adj} mobile phone - {feat}"
        ],
        "laptop": ["{adj} laptop for {use}", "{brand} {adj} notebook, {feat}", "{adj} ultrabook - {feat}"],
        "tablet": ["{adj} tablet with {feat}", "{brand} tablet - {feat}", "{adj} slate for {use}"],
        "sofa": ["{adj} sofa 3-seater", "{adj} couch with {feat}", "{adj} living room sofa, {feat}"],
        "chair": ["{adj} chair for {use}", "ergonomic {adj} chair", "{adj} dining chair"],
        "table": ["{adj} table for {use}", "{adj} coffee table", "{brand} {adj} table"],
        "tshirt": ["{adj} cotton t-shirt", "{brand} {adj} tee", "{adj} graphic t-shirt"],
        "jeans": ["{adj} jeans slim fit", "{brand} denim jeans", "{adj} blue jeans"],
        "jacket": ["{adj} jacket with {feat}", "{brand} {adj} jacket", "{adj} winter jacket - {feat}"]
    }
    adjectives = ["lightweight", "durable", "stylish", "classic", "premium"]
    brands = ["Acme", "Orion", "Nova", "Helix"]
    feats = ["64GB storage", "water-resistant", "wireless charging", "memory foam"]
    uses = ["office", "gaming", "travel", "dining"]

    rows = []
    per_leaf = 120  # enough samples to be stable
    for parent, leaves in taxonomy.items():
        for leaf in leaves:
            for i in range(per_leaf):
                tpl = rng.choice(templates[leaf])
                desc = tpl.format(adj=rng.choice(adjectives), brand=rng.choice(brands),
                                  feat=rng.choice(feats), use=rng.choice(uses))
                # add small variant punctuation / casing noise
                if rng.rand() < 0.2:
                    desc = desc.upper()
                if rng.rand() < 0.15:
                    desc += " - new"
                rows.append({"description": desc, "category": leaf, "parent": parent})
    df = pd.DataFrame(rows)
    # Shuffle deterministically
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df, taxonomy


def _sibling_confusion_from_cm(cm, labels, y_true, y_pred):
    # mapping for this synthetic taxonomy
    parent_map = {
        "smartphone": "Electronics", "laptop": "Electronics", "tablet": "Electronics",
        "sofa": "Home", "chair": "Home", "table": "Home",
        "tshirt": "Apparel", "jeans": "Apparel", "jacket": "Apparel"
    }
    n = len(labels)
    fractions = []
    for i, lab in enumerate(labels):
        total = cm[i].sum()
        if total <= 0:
            continue
        errors = total - cm[i, i]
        if errors == 0:
            fractions.append(0.0)
            continue
        sib_err = 0
        for j, lab2 in enumerate(labels):
            if j == i:
                continue
            if parent_map.get(lab2) == parent_map.get(lab):
                sib_err += cm[i, j]
        fractions.append(sib_err / errors)
    return float(np.mean(fractions)) if fractions else 0.0


def run(seed: int = 42) -> Dict:
    """Run the product categorisation experiment.

    Returns a JSON-serialisable dict with keys:
      - macro_f1: macro F1 score on held-out set (float)
      - unknown_rate: fraction of held-out assigned to UNKNOWN due to low confidence (float)
      - sibling_confusion: average fraction of errors that go to sibling categories (float)
      - top_k_example: a small example of top-3 candidates for the first held-out sample (list)
    """
    df, taxonomy = _make_synthetic_catalog(seed)
    X = df["description"].values
    y = df["category"].values
    parents = df["parent"].values

    X_train, X_test, y_train, y_test, p_train, p_test = train_test_split(
        X, y, parents, test_size=0.25, stratify=y, random_state=seed
    )

    # min_df lowered to 1 to ensure features are captured reliably on small splits
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1)
    Xtr = vec.fit_transform(X_train)
    Xte = vec.transform(X_test)

    # some sklearn versions don't accept multi_class argument in constructor used here
    # rely on solver default and remove multi_class to be compatible
    clf = LogisticRegression(max_iter=400, random_state=seed)
    clf.fit(Xtr, y_train)

    probs = clf.predict_proba(Xte)
    classes = clf.classes_
    # threshold for low-confidence
    threshold = 0.60
    maxp = probs.max(axis=1)
    pred_idx = probs.argmax(axis=1)
    preds = [classes[i] if p >= threshold else "UNKNOWN" for i, p in zip(pred_idx, maxp)]

    # For top-k candidates return list of (label, prob)
    def top_k_for_row(row_probs, k=3):
        idxs = np.argsort(row_probs)[::-1][:k]
        return [(classes[i], float(row_probs[i])) for i in idxs]

    top3_example = top_k_for_row(probs[0], k=3)

    # Compute macro F1 ignoring UNKNOWN (we score only on known categories; treat UNKNOWN as wrong)
    y_pred_for_score = [p if p != "UNKNOWN" else "__UNK__" for p in preds]
    macro = f1_score(y_test, y_pred_for_score, labels=list(classes), average="macro")

    unknown_rate = float(np.mean([1 if p == "UNKNOWN" else 0 for p in preds]))

    sibling_conf = _sibling_confusion_from_cm(
        confusion_matrix(y_test, [p if p != "UNKNOWN" else "__UNK__" for p in preds], labels=list(classes)),
        classes, y_test, preds)

    return {
        "macro_f1": float(macro),
        "unknown_rate": float(unknown_rate),
        "sibling_confusion": float(sibling_conf),
        "top_k_example": top3_example
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
