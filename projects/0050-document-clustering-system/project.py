import json
from typing import Dict, List
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity

def _build_synthetic_corpus() -> List[str]:
    tech = [
        "cloud service deployment and container orchestration",
        "container orchestration with Kubernetes for cloud apps",
        "microservices, containers and cloud deployment patterns",
    ]
    health = [
        "cardiovascular health and dietary recommendations",
        "preventive care for heart disease and healthy diet",
        "heart health diet and exercise suggestions",
    ]
    finance = [
        "stock market analysis and portfolio diversification",
        "investment portfolio and risk management for stocks",
        "diversification strategies for long term investment",
    ]
    travel = [
        "budget travel planning and local accommodation tips",
        "travel itineraries for short city breaks on a budget",
        "local guides and budget-friendly travel planning",
    ]
    extras = [
        "cloud apps and container deployment",
        "cardio diet suggestions",
        "portfolio diversification strategies",
        "city travel budget tips",
    ]
    docs = tech + health + finance + travel + extras
    docs.append("cloud apps and container deployment")
    return docs

def _deduplicate(docs: List[str]) -> List[str]:
    seen = set(); unique = []
    for d in docs:
        key = d.strip().lower()
        if key not in seen:
            seen.add(key); unique.append(d)
    return unique

def _top_terms_per_cluster(vectorizer: TfidfVectorizer, X, labels, top_n: int = 5) -> Dict[int, List[str]]:
    terms = np.array(vectorizer.get_feature_names_out())
    clusters = {}
    for k in np.unique(labels):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            clusters[int(k)] = []
            continue
        cent = X[idx].mean(axis=0)
        arr = np.asarray(cent).ravel() if hasattr(cent, 'A1') else cent.ravel()
        top_idx = arr.argsort()[::-1][:top_n]
        clusters[int(k)] = terms[top_idx].tolist()
    return clusters

def _similarity_baseline_grouping(X, docs: List[str], threshold: float = 0.5) -> List[int]:
    sim = cosine_similarity(X)
    groups = [-1] * len(docs); next_group = 0
    for i in range(len(docs)):
        if groups[i] != -1: continue
        groups[i] = next_group
        for j in range(i + 1, len(docs)):
            if groups[j] == -1 and sim[i, j] >= threshold:
                groups[j] = next_group
        next_group += 1
    return groups

def run(seed: int = 42) -> Dict:
    docs = _build_synthetic_corpus()
    docs = _deduplicate(docs)
    train_docs, hold_docs = train_test_split(docs, test_size=0.3, random_state=seed)
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
    X_train = vectorizer.fit_transform(train_docs)
    X_hold = vectorizer.transform(hold_docs)
    k = 4
    km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X_train)
    labels_train = km.labels_
    centers = km.cluster_centers_
    sim_to_centres = cosine_similarity(X_hold, centers)
    hold_labels = sim_to_centres.argmax(axis=1)
    sil = None
    try:
        if len(np.unique(hold_labels)) >= 2 and len(hold_labels) > len(np.unique(hold_labels)):
            sil = float(silhouette_score(X_hold, hold_labels, metric='cosine'))
        else:
            sil = float('nan')
    except Exception:
        sil = float('nan')
    top_terms = _top_terms_per_cluster(vectorizer, X_train, labels_train, top_n=5)
    sizes = {int(k): int((labels_train == k).sum()) for k in np.unique(labels_train)}
    X_all = vectorizer.transform(docs)
    baseline_labels = _similarity_baseline_grouping(X_all, docs, threshold=0.45)
    baseline_sizes = {}
    for b in baseline_labels:
        baseline_sizes[int(b)] = baseline_sizes.get(int(b), 0) + 1
    km_other = KMeans(n_clusters=k, random_state=seed + 1, n_init=10).fit(X_train)
    labels_other = km_other.labels_
    def label_agreement(a: np.ndarray, b: np.ndarray) -> float:
        if len(a) != len(b): return 0.0
        return float((a == b).sum()) / float(len(a))
    stability = label_agreement(labels_train, labels_other)
    editable_labels = {int(k): f"Cluster {k}" for k in np.unique(labels_train)}
    move_action = {"doc": train_docs[0], "from_cluster": int(labels_train[0]), "to_cluster": 0}
    result = {
        "n_documents": len(docs),
        "n_train": len(train_docs),
        "n_hold": len(hold_docs),
        "k": k,
        "silhouette": sil if not np.isnan(sil) else None,
        "stability_agreement": float(stability),
        "cluster_sizes": sizes,
        "top_terms": top_terms,
        "baseline_group_sizes": baseline_sizes,
        "editable_labels": editable_labels,
        "example_move_action": move_action,
        "seed": int(seed),
    }
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
