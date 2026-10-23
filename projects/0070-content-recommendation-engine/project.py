import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _make_data(seed: int = 42):
    rng = np.random.RandomState(seed)
    topics = ["science", "arts", "tech", "business", "health"]
    items = []
    for i in range(20):
        items.append({"item_id": f"I{i}", "topic": rng.choice(topics), "days_ago": int(rng.randint(0, 366))})
    items_df = pd.DataFrame(items)
    users = [f"U{j}" for j in range(8)]
    ratings = []
    for u in users:
        n = rng.randint(4, 11)
        rated = rng.choice(items_df["item_id"].values, size=n, replace=False)
        for it in rated:
            ratings.append({"user_id": u, "item_id": it, "rating": float(rng.randint(1, 6))})
    ratings_df = pd.DataFrame(ratings)
    return items_df, ratings_df


def _build_user_item_matrix(items_df: pd.DataFrame, ratings_df: pd.DataFrame):
    mat = ratings_df.pivot_table(index="user_id", columns="item_id", values="rating")
    return mat.reindex(columns=items_df["item_id"].tolist())


def _freshness_boost(days_ago: np.ndarray):
    return np.exp(-days_ago / 180.0)


def _diverse_rerank(candidates: List[str], scores: Dict[str, float], topics: Dict[str, str], top_k: int = 5, lambda_div: float = 0.7):
    selected = []
    covered = set()
    candidates = sorted(candidates, key=lambda i: -scores.get(i, 0.0))
    for _ in range(min(top_k, len(candidates))):
        best, best_val = None, -1e9
        for c in candidates:
            if c in selected:
                continue
            score = scores.get(c, 0.0)
            diversity_penalty = 1.0 if topics.get(c) in covered else 0.0
            val = (1 - lambda_div) * score - lambda_div * diversity_penalty
            if val > best_val:
                best_val, best = val, c
        if best is None:
            break
        selected.append(best)
        covered.add(topics.get(best))
    return selected


def run(seed: int = 42) -> Dict:
    items_df, ratings_df = _make_data(seed)
    user_item = _build_user_item_matrix(items_df, ratings_df)

    # hold out one rating per user
    train_rows, holdout_rows = [], []
    rng = np.random.RandomState(seed)
    for u in user_item.index:
        rated = user_item.loc[u].dropna().index.tolist()
        if not rated:
            continue
        hold = rng.choice(rated)
        for it in rated:
            row = {"user_id": u, "item_id": it, "rating": user_item.at[u, it]}
            (holdout_rows if it == hold else train_rows).append(row)
    train_df = pd.DataFrame(train_rows)
    holdout_df = pd.DataFrame(holdout_rows)

    train_mat = train_df.pivot_table(index="user_id", columns="item_id", values="rating")
    train_mat = train_mat.reindex(index=user_item.index, columns=user_item.columns)
    imputed = train_mat.astype(float)
    user_means = imputed.mean(axis=1)
    imputed = imputed.apply(lambda row: row.fillna(user_means[row.name]), axis=1).fillna(0.0)
    sim = cosine_similarity(imputed.values)
    user_index = list(imputed.index)
    user_to_idx = {u: i for i, u in enumerate(user_index)}

    item_topics = dict(zip(items_df["item_id"], items_df["topic"]))
    item_fresh = dict(zip(items_df["item_id"], _freshness_boost(items_df["days_ago"].values)))

    def _baseline_recommend(top_k=5, train_df_local=None):
        td = train_df_local if train_df_local is not None else train_df
        pop = td.groupby("item_id").size().reindex(user_item.columns).fillna(0.0)
        scores = {it: pop.get(it, 0.0) * (1.0 + 0.3 * item_fresh.get(it, 0.0)) for it in user_item.columns}
        return sorted(scores.keys(), key=lambda i: -scores[i])[:top_k]

    def recommend_for(user_id: str, top_k: int = 5):
        if user_id not in user_to_idx:
            return _baseline_recommend(top_k)
        idx = user_to_idx[user_id]
        sims = sim[idx].copy()
        sims[idx] = -1
        neighbor_idx = int(np.argmax(sims))
        if sims[neighbor_idx] <= 0:
            return _baseline_recommend(top_k)
        neighbor = user_index[neighbor_idx]
        seen = set(train_mat.loc[user_id].dropna().index.tolist())
        candidates = [it for it in user_item.columns if it not in seen]

        # compute popularity normalized to combine with neighbor signal
        pop = train_df.groupby("item_id").size().reindex(user_item.columns).fillna(0.0)
        pop_norm = (pop / (pop.max() + 1e-9)).to_dict()

        scores = {}
        for it in candidates:
            r = train_mat.at[neighbor, it] if it in train_mat.columns and not pd.isna(train_mat.at[neighbor, it]) else 0.0
            # combine neighbor rating signal with popularity to increase chance of hits
            neighbor_score = float(r) * (1.0 + 0.5 * item_fresh.get(it, 0.0))
            pop_score = float(pop_norm.get(it, 0.0)) * (1.0 + 0.3 * item_fresh.get(it, 0.0))
            scores[it] = 0.6 * neighbor_score + 0.4 * pop_score
        if all(v == 0.0 for v in scores.values()):
            return _baseline_recommend(top_k)
        return _diverse_rerank(candidates, scores, item_topics, top_k=top_k)

    def recall_at_k(k=5):
        hits = total = 0
        for _, row in holdout_df.iterrows():
            recs = recommend_for(row["user_id"], top_k=k)
            if row["item_id"] in recs:
                hits += 1
            total += 1
        return hits / max(1, total)

    recall5 = recall_at_k(5)
    coverage = float(np.mean([
        (len({item_topics[it] for it in recommend_for(u, top_k=5)}) / max(1, min(5, len(recommend_for(u, top_k=5)))))
        for u in user_item.index
    ]))

    # New-user cold-start: hide one user and test baseline
    new_user = user_item.index[0]
    train_df_nu = train_df[train_df["user_id"] != new_user]
    baseline_top = _baseline_recommend(top_k=5, train_df_local=train_df_nu)
    hold_new = holdout_df[holdout_df["user_id"] == new_user]
    new_user_recall = 1.0 if (not hold_new.empty and hold_new.iloc[0]["item_id"] in baseline_top) else 0.0

    baseline_r5 = 0.0
    if not holdout_df.empty:
        baseline_r5 = sum(1 for _, r in holdout_df.iterrows() if r["item_id"] in _baseline_recommend(5)) / len(holdout_df)

    sample_user = user_item.index[1]
    sample_recs = recommend_for(sample_user, top_k=5)

    return {
        "recall_at_5": float(recall5),
        "topic_coverage_at_5": float(coverage),
        "new_user_recall": float(new_user_recall),
        "baseline_recall_at_5": float(baseline_r5),
        "sample_user": str(sample_user),
        "sample_recommendations": list(sample_recs),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
