import json
from typing import Dict
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity


def make_synthetic_embeddings(n_identities=12, per_identity=8, emb_dim=128, session_shift=0.05, unknown_ratio=0.25, random_state=42):
    rng = np.random.RandomState(random_state)
    prototypes = rng.normal(scale=1.0, size=(n_identities, emb_dim))
    # First session embeddings
    session1 = []
    labels1 = []
    for i in range(n_identities):
        samples = prototypes[i] + rng.normal(scale=0.2, size=(per_identity, emb_dim))
        session1.append(samples)
        labels1 += [i] * per_identity
    session1 = np.vstack(session1)
    labels1 = np.array(labels1)
    # Second session: shift prototypes to simulate temporal drift
    prototypes2 = prototypes + rng.normal(scale=session_shift, size=prototypes.shape)
    session2 = []
    labels2 = []
    for i in range(n_identities):
        samples = prototypes2[i] + rng.normal(scale=0.25, size=(per_identity, emb_dim))
        session2.append(samples)
        labels2 += [i] * per_identity
    session2 = np.vstack(session2)
    labels2 = np.array(labels2)
    # Unknown embeddings (not from any identity)
    n_unknown = int(unknown_ratio * session2.shape[0])
    unknown = rng.normal(scale=1.0, size=(n_unknown, emb_dim))
    return session1, labels1, session2, labels2, unknown


def identify_gallery_probe(gallery_embeddings, gallery_labels, probe_embeddings, threshold=0.5):
    # Cosine similarity based nearest neighbour with threshold for unknown
    sims = cosine_similarity(probe_embeddings, gallery_embeddings)
    idx = sims.argmax(axis=1)
    best_sims = sims.max(axis=1)
    pred_labels = gallery_labels[idx]
    # Apply threshold: below threshold -> unknown (-1)
    pred_labels[best_sims < threshold] = -1
    return pred_labels, best_sims


def compute_rates(true_labels, pred_labels, unknown_label=-1):
    # true_labels may include unknowns labelled as -1
    assert true_labels.shape == pred_labels.shape
    # For identification accuracy among known probes only
    known_mask = true_labels != unknown_label
    if known_mask.sum() == 0:
        top1 = float('nan')
    else:
        top1 = float((pred_labels[known_mask] == true_labels[known_mask]).mean())
    # False Match Rate (FMR): when unknown probe is incorrectly matched to a gallery identity
    unknown_mask = true_labels == unknown_label
    fmr = float((pred_labels[unknown_mask] != unknown_label).mean()) if unknown_mask.sum() > 0 else 0.0
    # False Non-Match Rate (FNMR): when a known probe is rejected (predicted unknown)
    fnmr = float((pred_labels[known_mask] == unknown_label).mean()) if known_mask.sum() > 0 else 0.0
    # Unknown rejection rate: proportion of unknown probes correctly rejected
    unknown_reject = float((pred_labels[unknown_mask] == unknown_label).mean()) if unknown_mask.sum() > 0 else 0.0
    return {'top1_accuracy': top1, 'false_match_rate': fmr, 'false_non_match_rate': fnmr, 'unknown_reject_rate': unknown_reject}


def run(seed: int = 42) -> Dict[str, float]:
    # Create synthetic embeddings for two sessions
    session1, labels1, session2, labels2, unknown = make_synthetic_embeddings(random_state=seed)
    # Build a gallery from session1 (enrolment) and probes from session2 plus unknowns
    # For simplicity use one embedding per identity in the gallery: mean prototype from session1
    df_gallery = pd.DataFrame(session1)
    df_gallery['label'] = labels1
    gallery_embeddings = df_gallery.groupby('label').mean().values
    gallery_labels = df_gallery.groupby('label')['label'].first().values.astype(int)
    # Probes: all session2 embeddings plus true unknowns
    probe_embeddings = np.vstack([session2, unknown])
    probe_labels = np.concatenate([labels2, np.full(unknown.shape[0], -1, dtype=int)])
    # Split probes into held-out test set to simulate unseen probes
    pe_train, pe_test, pl_train, pl_test = train_test_split(probe_embeddings, probe_labels, test_size=0.5, random_state=seed, stratify=(probe_labels != -1))
    # We will evaluate on pe_test
    # Calibrate a simple threshold on the small pe_train set: choose threshold that balances FMR and FNMR
    # Evaluate a range and pick threshold minimising abs(FMR-FNMR)
    thresholds = np.linspace(0.2, 0.9, 36)
    best_thr = 0.5
    best_score = 1e9
    for thr in thresholds:
        preds, sims = identify_gallery_probe(gallery_embeddings, gallery_labels, pe_train, threshold=thr)
        rates = compute_rates(pl_train, preds)
        score = abs(rates['false_match_rate'] - rates['false_non_match_rate'])
        if score < best_score:
            best_score = score
            best_thr = float(thr)
    # Final evaluation on held-out test probes
    preds_test, sims_test = identify_gallery_probe(gallery_embeddings, gallery_labels, pe_test, threshold=best_thr)
    rates = compute_rates(pl_test, preds_test)
    # Return the main metrics and a few extras
    out = {
        'top1_accuracy': float(rates['top1_accuracy']),
        'false_match_rate': float(rates['false_match_rate']),
        'false_non_match_rate': float(rates['false_non_match_rate']),
        'unknown_reject_rate': float(rates['unknown_reject_rate']),
        'threshold': float(best_thr),
        'gallery_size': int(gallery_embeddings.shape[0])
    }
    return out


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
