import json
from typing import Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def _build_ratings() -> pd.DataFrame:
    data = {
        "user_id": ["u1","u1","u1","u2","u2","u3","u3","u3","u4","u4","u5","u5"],
        "item_id": ["i1","i2","i3","i2","i4","i1","i3","i5","i2","i5","i3","i4"],
        "rating":  [5.0, 4.0, 1.0, 5.0, 3.0, 4.0, 2.0, 5.0, 3.0, 4.0, 1.0, 2.0],
    }
    return pd.DataFrame(data)

def _pivot(r: pd.DataFrame) -> pd.DataFrame:
    return r.pivot_table(index="user_id", columns="item_id", values="rating")

def _holdout(ratings: pd.DataFrame, seed: int = 42, hold_per_user: int = 1):
    train_idx = []
    test_idx = []
    rng = np.random.RandomState(seed)
    for _, g in ratings.groupby("user_id"):
        idx = g.index.tolist()
        if len(idx) <= hold_per_user:
            train_idx.extend(idx)
            continue
        held = rng.choice(idx, size=hold_per_user, replace=False).tolist()
        for i in idx:
            (test_idx if i in held else train_idx).append(i)
    return ratings.loc[train_idx].reset_index(drop=True), ratings.loc[test_idx].reset_index(drop=True)

def _user_sim(mat: pd.DataFrame) -> pd.DataFrame:
    M = mat.fillna(0).values
    sim = cosine_similarity(M)
    return pd.DataFrame(sim, index=mat.index, columns=mat.index)

def _popularity(train: pd.DataFrame) -> List[str]:
    agg = train.groupby("item_id").agg(count=("rating","size"), mean=("rating","mean"))
    agg = agg.sort_values(["count","mean"], ascending=[False,False])
    return agg.index.tolist()

def _rec_for_user(target: str, ui: pd.DataFrame, sim: pd.DataFrame, train: pd.DataFrame, top_k=3, neigh_n=2) -> List[str]:
    # popularity list for filling recommendations
    pop = _popularity(train)
    if target not in ui.index:
        return [i for i in pop if i not in []][:top_k]
    sims = sim.loc[target].drop(labels=[target])
    topn = sims[sims>0].sort_values(ascending=False).head(neigh_n)
    if topn.empty:
        # cold or no positive neighbors: return top popularity
        return [i for i in pop if i not in []][:top_k]
    unseen = ui.loc[target][ui.loc[target].isna()].index.tolist()
    if not unseen:
        return []
    scores = {}
    for neigh, w in topn.items():
        for item, r in ui.loc[neigh].items():
            if item in unseen and not pd.isna(r):
                # accumulate weighted rating
                scores[item] = scores.get(item,0.0) + w * float(r)
    # if no scored items, fallback to popularity
    if not scores:
        return [i for i in pop if i in unseen][:top_k]
    # sort scored items
    ranked = [i for i,_ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]
    # if not enough, fill with popular unseen items
    for p in pop:
        if len(ranked) >= top_k:
            break
        if p in unseen and p not in ranked:
            ranked.append(p)
    return ranked[:top_k]

def _prec_at_k(recs: List[str], rel: List[str], k=3) -> float:
    if not recs: return 0.0
    recs_k = recs[:k]
    hits = sum(1 for r in recs_k if r in rel)
    return hits / float(min(k, len(recs_k)))

def _coverage(all_recs: List[List[str]], catalog_size: int) -> float:
    uniq = set().union(*[set(r) for r in all_recs]) if all_recs else set()
    return len(uniq) / float(catalog_size) if catalog_size else 0.0

def run(seed: int = 42) -> Dict:
    ratings = _build_ratings()
    train, test = _holdout(ratings, seed=seed, hold_per_user=1)
    ui = _pivot(train)
    ui = ui.sort_index().reindex(sorted(ui.columns), axis=1)
    sim = _user_sim(ui)
    pop = _popularity(train)
    users = test["user_id"].unique().tolist()
    per_prec = []
    per_base = []
    all_recs = []
    all_base = []
    # increase neighbor count to improve recall/precision on this small dataset
    neigh_n = 4
    for u in users:
        rel = test[test["user_id"]==u]["item_id"].tolist()
        recs = _rec_for_user(u, ui, sim, train, top_k=3, neigh_n=neigh_n)
        base = pop[:3]
        all_recs.append(recs)
        all_base.append(base)
        per_prec.append(_prec_at_k(recs, rel, k=3))
        per_base.append(_prec_at_k(base, rel, k=3))
    catalog = ratings["item_id"].nunique()
    cov = _coverage(all_recs, catalog)
    base_cov = _coverage(all_base, catalog)
    mean_prec = float(np.mean(per_prec)) if per_prec else 0.0
    mean_base = float(np.mean(per_base)) if per_base else 0.0
    cold = _rec_for_user("new_user", ui, sim, train, top_k=3, neigh_n=neigh_n)
    return {
        "precision_at_3": round(mean_prec,4),
        "baseline_precision_at_3": round(mean_base,4),
        "coverage": round(cov,4),
        "baseline_coverage": round(base_cov,4),
        "example_recommendation_for_u1": json.dumps(_rec_for_user("u1", ui, sim, train, top_k=3, neigh_n=neigh_n)),
        "cold_start_recommendation": json.dumps(cold),
    }

if __name__ == "__main__":
    import json as _json
    print(_json.dumps(run(42), indent=2))
