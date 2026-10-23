import json
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize


def tokenize(text: str) -> List[str]:
    # Simple whitespace + punctuation split (teaching-friendly, deterministic)
    for ch in ".,;:!?()[]":
        text = text.replace(ch, " ")
    tokens = [t.lower() for t in text.split() if t.strip()]
    return tokens


def build_corpus() -> List[str]:
    # Small synthetic corpus created for teaching/demonstration purposes.
    sents = [
        "The cat sat on the mat",
        "The dog chased the cat",
        "A child plays with the dog",
        "Many children like the school and the teacher",
        "The teacher helps the student",
        "Students read books at school",
        "A doctor works at the hospital",
        "The nurse assists the doctor",
        "The hospital has many nurses and doctors",
        "Birds can fly in the sky",
        "A pet cat and a pet dog live in the house",
        "The city has a big hospital and a small school",
    ]
    # Return as one text (tokeniser is simple)
    return sents


def build_vocab(tokens: List[str], min_freq: int = 2) -> Tuple[Dict[str, int], List[str]]:
    freq = pd.Series(tokens).value_counts()
    vocab = [w for w, c in freq.items() if c >= min_freq]
    vocab = sorted(vocab)
    stoi = {w: i for i, w in enumerate(vocab)}
    return stoi, vocab


def cooccurrence_matrix(tokens: List[str], stoi: Dict[str, int], window: int = 2) -> np.ndarray:
    V = len(stoi)
    M = np.zeros((V, V), dtype=float)
    N = len(tokens)
    for i, w in enumerate(tokens):
        if w not in stoi:
            continue
        iw = stoi[w]
        left = max(0, i - window)
        right = min(N, i + window + 1)
        for j in range(left, right):
            if j == i:
                continue
            w2 = tokens[j]
            if w2 not in stoi:
                continue
            jw = stoi[w2]
            M[iw, jw] += 1.0
    return M


def pmi_matrix(co: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    total = co.sum()
    if total == 0:
        return co
    p_wc = co / total
    p_w = p_wc.sum(axis=1)
    p_c = p_wc.sum(axis=0)
    # Outer product
    expected = np.outer(p_w, p_c)
    with np.errstate(divide='ignore'):
        pmi = np.log((p_wc + eps) / (expected + eps))
    pmi[pmi < 0] = 0.0
    return pmi


def nearest_neighbors(emb: np.ndarray, vocab: List[str], query: str, topk: int = 5) -> List[Tuple[str, float]]:
    if query not in vocab:
        return []
    idx = vocab.index(query)
    vecs = emb
    q = vecs[idx:idx+1]
    sims = (vecs @ q.T).ravel()
    norms = np.linalg.norm(vecs, axis=1) * np.linalg.norm(q)
    norms[norms == 0] = 1e-8
    sims = sims / norms
    # remove self
    sims[idx] = -np.inf
    top_idx = np.argsort(-sims)[:topk]
    return [(vocab[i], float(sims[i])) for i in top_idx]


def run(seed: int = 42) -> Dict:
    np.random.seed(int(seed))
    sents = build_corpus()
    tokens = []
    for s in sents:
        tokens.extend(tokenize(s))
    stoi, vocab = build_vocab(tokens, min_freq=2)
    # use a slightly larger window to capture more context in this small corpus
    co = cooccurrence_matrix(tokens, stoi, window=4)
    pmi = pmi_matrix(co)
    # Truncated SVD as a simple embedding method (teaching-friendly, deterministic via seed)
    # choose more dimensions (up to a reasonable cap) to better capture relations in tiny vocabularies
    dim = min(30, max(2, len(vocab) - 1))
    if dim < 2:
        dim = 2
    svd = TruncatedSVD(n_components=dim, random_state=int(seed))
    emb = svd.fit_transform(pmi)
    emb = normalize(emb, axis=1)

    # Small gold standard: expected related words per query for a simple check
    gold = {
        "cat": {"dog", "pet", "animal"},
        "teacher": {"student", "school"},
        "doctor": {"nurse", "hospital"},
    }

    # Compute neighbourhood quality metric: fraction of top-3 neighbours that are in the gold set
    qualities = {}
    for q, related in gold.items():
        if q not in vocab:
            qualities[q] = 0.0
            continue
        nbs = nearest_neighbors(emb, vocab, q, topk=3)
        if not nbs:
            qualities[q] = 0.0
            continue
        found = sum(1 for w, _ in nbs if w in related)
        qualities[q] = float(found / 3.0)

    # Main metric: average quality across the gold queries
    main_metric = float(np.mean(list(qualities.values())) if qualities else 0.0)

    # Example explorer output for a few queries (deterministic)
    explorer = {}
    for q in ["cat", "teacher", "doctor", "school"]:
        explorer[q] = nearest_neighbors(emb, vocab, q, topk=5)

    result = {
        "vocab_size": int(len(vocab)),
        "embedding_dim": int(emb.shape[1]),
        "neighbour_quality": qualities,
        "main_metric": main_metric,
        "explorer": explorer,
        "provenance": "synthetic in-code corpus; tokenisation=whitespace+punctuation; min_freq=2; model=PMI+SVD approximation",
    }
    return result


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
