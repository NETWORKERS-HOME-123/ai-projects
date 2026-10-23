import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def _make_reference_caption(label: int, image: np.ndarray) -> str:
    # Simple synthetic reference caption based on true label and a crude fill attribute
    fill = 'filled' if image.mean() > 4.0 else 'stroked'
    return f'A handwritten digit {label} that is {fill}.'


def _decode_caption(pred_label: int, image: np.ndarray) -> str:
    # Deterministic template decoder from predicted label and a visible attribute
    fill = 'filled' if image.mean() > 4.0 else 'stroked'
    return f'A handwritten digit {pred_label} that is {fill}.'


def _token_f1_score(references, predictions):
    # Micro-averaged token-level F1 between token sets of reference and prediction
    tp = 0
    fp = 0
    fn = 0
    for ref, pred in zip(references, predictions):
        rset = set(str(ref).lower().split())
        pset = set(str(pred).lower().split())
        tp += len(rset & pset)
        fp += len(pset - rset)
        fn += len(rset - pset)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(f1)


def run(seed: int = 42) -> Dict[str, object]:
    # Load data
    digits = load_digits()
    X = digits.data  # (n_samples, 64)
    y = digits.target

    # Build reference captions (synthetic)
    refs = [_make_reference_caption(int(lbl), img.reshape(8, 8)) for lbl, img in zip(y, digits.images)]

    # Split with reproducibility
    X_train, X_test, y_train, y_test, refs_train, refs_test, imgs_train, imgs_test = train_test_split(
        X, y, refs, digits.images, test_size=0.25, random_state=seed, stratify=y
    )

    # Simple image encoder: RandomForest on flattened pixels
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)

    # Build predicted captions via deterministic decoder
    pred_caps = [_decode_caption(int(p), img) for p, img in zip(preds, imgs_test)]

    # Evaluation: digit accuracy and token-level F1 for captions
    digit_acc = float(accuracy_score(y_test, preds))
    caption_token_f1 = _token_f1_score(refs_test, pred_caps)

    out = {
        'digit_accuracy': digit_acc,
        'caption_token_f1': caption_token_f1,
        'num_train': int(X_train.shape[0]),
        'num_test': int(X_test.shape[0]),
        'model': 'RandomForestClassifier',
        'decoder': 'deterministic-template',
    }
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
