import json
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def generate_synthetic(seed: int = 42):
    rng = np.random.default_rng(seed)
    n_users, n_items = 200, 200
    n_clusters, items_per_cluster = 20, 10
    items = pd.DataFrame({"item_id": np.arange(n_items), "in_stock": np.ones(n_items, dtype=bool)})
    rows = []
    for u in range(n_users):
        c = u % n_clusters
        offset = u // n_clusters
        base = c * items_per_cluster
        seq = [base + ((offset + i) % items_per_cluster) for i in range(6)]
        # tiny deterministic jitter in lengths keeps users slightly different while preserving structure
        if rng.random() < 0.25:
            seq = seq[:-1]
        for t, it in enumerate(seq):
            rows.append((u, int(it), 1, int(t)))
    interactions = pd.DataFrame(rows, columns=["user_id", "item_id", "rating", "time_idx"])
    return interactions, items


def train_test_chrono_split(interactions: pd.DataFrame):
    interactions = interactions.sort_values(["user_id", "time_idx"]).copy()
    held_out = interactions.groupby("user_id", sort=False).tail(1)
    train = interactions.drop(index=held_out.index)
    return train.reset_index(drop=True), held_out.reset_index(drop=True)


def build_user_item_matrix(train: pd.DataFrame, n_users: int, n_items: int):
    mat = np.zeros((n_users, n_items), dtype=float)
    for r in train.itertuples(index=False):
        mat[int(r.user_id), int(r.item_id)] = 1.0
    return mat


class SimpleOptOutWidget:
    def __init__(self):
        self.dismissed_users = set()

    def dismiss_for(self, user_id: int):
        self.dismissed_users.add(int(user_id))

    def reset(self, user_id: int):
        self.dismissed_users.discard(int(user_id))

    def is_opted_out(self, user_id: int) -> bool:
        return int(user_id) in self.dismissed_users


def recommend_for_user(
    user_id: int,
    user_item_mat: np.ndarray,
    similarity: np.ndarray,
    item_popularity: np.ndarray,
    items_in_stock: np.ndarray,
    widget: SimpleOptOutWidget,
    top_k: int = 5,
) -> List[int]:
    if widget.is_opted_out(user_id):
        return []
    user_vec = user_item_mat[user_id]
    seen = user_vec.astype(bool)
    neigh_scores = similarity[user_id] @ user_item_mat
    sim_sum = similarity[user_id].sum()
    if sim_sum > 0:
        neigh_scores = neigh_scores / sim_sum
    scores = item_popularity.copy() if user_vec.sum() == 0 or np.all(neigh_scores == 0) else 0.9 * neigh_scores + 0.1 * item_popularity
    scores[seen] = -np.inf
    scores[~items_in_stock] = -np.inf
    top_items = np.argsort(-scores)[:top_k]
    return [int(i) for i in top_items if np.isfinite(scores[i])]


def evaluate(train: pd.DataFrame, held_out: pd.DataFrame, items: pd.DataFrame, seed: int = 42) -> Dict:
    n_users = int(max(train.user_id.max(), held_out.user_id.max())) + 1
    n_items = int(items.item_id.max()) + 1
    user_item = build_user_item_matrix(train, n_users, n_items)
    similarity = cosine_similarity(user_item)
    item_pop = train.groupby("item_id").size().reindex(range(n_items), fill_value=0).values.astype(float)
    item_pop = item_pop / item_pop.sum() if item_pop.sum() > 0 else np.ones(n_items) / n_items
    items_in_stock = items.sort_values("item_id").in_stock.values

    widget = SimpleOptOutWidget()
    rng = np.random.default_rng(seed)
    dismissed = rng.choice(n_users, size=max(1, n_users // 50), replace=False)
    for d in dismissed:
        widget.dismiss_for(int(d))

    top_k = 5
    hits = 0
    pop_hits = 0
    recommended_items_set = set()
    pop_scores = item_pop.copy()
    pop_scores[~items_in_stock] = -np.inf
    pop_ranking = np.argsort(-pop_scores)

    for r in held_out.itertuples(index=False):
        u = int(r.user_id)
        true_item = int(r.item_id)
        recs = recommend_for_user(u, user_item, similarity, item_pop, items_in_stock, widget, top_k=top_k)
        recommended_items_set.update(recs)
        if true_item in recs:
            hits += 1
        seen = set(np.where(user_item[u] > 0)[0])
        pop_list = [int(i) for i in pop_ranking if i not in seen][:top_k]
        if true_item in pop_list:
            pop_hits += 1

    n_eval = len(held_out[~held_out.user_id.isin(list(widget.dismissed_users))])
    recall_at_5 = float(hits / max(1, n_eval))
    popularity_recall_at_5 = float(pop_hits / max(1, n_eval))
    coverage = float(len(recommended_items_set) / max(1, n_items))

    users_with_history = np.where(user_item.sum(axis=1) > 0)[0]
    cold_user = int(users_with_history[0]) if len(users_with_history) > 0 else 0
    saved_row = user_item[cold_user].copy()
    user_item[cold_user] = 0
    cold_recs = recommend_for_user(cold_user, user_item, similarity, item_pop, items_in_stock, widget, top_k=top_k)
    user_item[cold_user] = saved_row
    cold_held = held_out[held_out.user_id == cold_user]
    new_user_recall = 1.0 if len(cold_held) > 0 and int(cold_held.iloc[0].item_id) in cold_recs else 0.0

    return {
        "recall_at_5": recall_at_5,
        "coverage": coverage,
        "popularity_recall_at_5": popularity_recall_at_5,
        "new_user_recall": float(new_user_recall),
    }


def run(seed: int = 42) -> Dict:
    interactions, items = generate_synthetic(seed)
    train, held_out = train_test_chrono_split(interactions)
    return evaluate(train, held_out, items, seed=seed)


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
