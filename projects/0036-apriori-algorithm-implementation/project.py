import json
import itertools
from typing import Iterable, Tuple, Set, FrozenSet, Dict, List

Transaction = FrozenSet[str]

def _build_transactions() -> List[Transaction]:
    # Tiny, hand-crafted basket collection with easy-to-count frequencies
    raw = [
        ["A", "B", "C"],
        ["A", "B"],
        ["A", "C"],
        ["B", "C"],
        ["A", "B", "D"],
        ["B", "D"],
        ["C", "D"],
    ]
    return [frozenset(x) for x in raw]


def _count_support(transactions: Iterable[Transaction], candidates: Iterable[Transaction]) -> Dict[Transaction, int]:
    counts: Dict[Transaction, int] = {c: 0 for c in candidates}
    for t in transactions:
        for c in counts:
            if c.issubset(t):
                counts[c] += 1
    return counts


def _join_step(prev_frequents: List[Transaction], k: int) -> List[Transaction]:
    # join step: combine pairs of (k-1)-itemsets to form k-item candidates
    candidates: Set[Transaction] = set()
    n = len(prev_frequents)
    for i in range(n):
        for j in range(i+1, n):
            a = prev_frequents[i]
            b = prev_frequents[j]
            union = a | b
            if len(union) == k:
                candidates.add(frozenset(union))
    return sorted(candidates)


def _prune_step(candidates: List[Transaction], prev_frequents_set: Set[Transaction]) -> List[Transaction]:
    # prune any candidate that has a (k-1)-subset not in previous frequents
    pruned: List[Transaction] = []
    for c in candidates:
        all_subsets_ok = True
        for subset in itertools.combinations(c, len(next(iter(prev_frequents_set))) if prev_frequents_set else len(c)-1):
            # More robust approach: check all (k-1) subsets
            break
        # Instead of the above fragile step, enumerate (k-1)-subsets directly
        k = len(c)
        for subset in itertools.combinations(c, k-1):
            if frozenset(subset) not in prev_frequents_set:
                all_subsets_ok = False
                break
        if all_subsets_ok:
            pruned.append(c)
    return sorted(pruned)


def apriori(transactions: List[Transaction], min_support: float) -> Tuple[Dict[int, List[Transaction]], Dict[Transaction, int], List[int]]:
    n_transactions = len(transactions)
    # first pass: count single items
    items = sorted({item for t in transactions for item in t})
    candidates1 = [frozenset([it]) for it in items]
    counts1 = _count_support(transactions, candidates1)
    frequents1 = [it for it in candidates1 if counts1[it] / n_transactions >= min_support]

    frequents_by_k: Dict[int, List[Transaction]] = {1: frequents1}
    support_counts: Dict[Transaction, int] = {**{k: v for k, v in counts1.items() if k in frequents1}}
    candidate_counts_per_iter: List[int] = [len(candidates1)]

    k = 2
    prev_frequents = frequents1
    while prev_frequents:
        candidates_k = _join_step(prev_frequents, k)
        candidate_counts_per_iter.append(len(candidates_k))
        prev_set = set(prev_frequents)
        candidates_k = _prune_step(candidates_k, prev_set)
        if not candidates_k:
            break
        counts_k = _count_support(transactions, candidates_k)
        frequents_k = [c for c in candidates_k if counts_k[c] / n_transactions >= min_support]
        for c in frequents_k:
            support_counts[c] = counts_k[c]
        if frequents_k:
            frequents_by_k[k] = frequents_k
        prev_frequents = frequents_k
        k += 1
    return frequents_by_k, support_counts, candidate_counts_per_iter


def exhaustive_frequent(transactions: List[Transaction], min_support: float) -> Set[Transaction]:
    n = len(transactions)
    universe = sorted({item for t in transactions for item in t})
    result: Set[Transaction] = set()
    for r in range(1, len(universe)+1):
        for comb in itertools.combinations(universe, r):
            combf = frozenset(comb)
            count = sum(1 for t in transactions if combf.issubset(t))
            if count / n >= min_support:
                result.add(combf)
    return result


def run(seed: int = 42) -> dict:
    # seed accepted to satisfy interface; algorithm is deterministic
    import numpy as _np
    _np.random.seed(seed)

    transactions = _build_transactions()
    n = len(transactions)
    # pick a pedagogical support threshold; small dataset needs coarse steps
    min_support = 0.4

    frequents_by_k, support_counts, candidate_counts = apriori(transactions, min_support)
    found = {it for lst in frequents_by_k.values() for it in lst}
    exhaustive = exhaustive_frequent(transactions, min_support)

    # agreement: fraction of frequent itemsets that match exhaustive enumeration (symmetric)
    if not found and not exhaustive:
        agreement = 1.0
    else:
        agreement = len(found & exhaustive) / len(found | exhaustive)

    result = {
        "agreement": float(agreement),
        "num_transactions": n,
        "min_support_used": float(min_support),
        "num_frequent_itemsets": int(len(found)),
        "max_candidate_count": int(max(candidate_counts) if candidate_counts else 0),
        # human-readable iteration details for review
        "iterations": [
            {"iteration": i+1, "candidate_count": int(c)} for i, c in enumerate(candidate_counts)
        ],
    }
    return result

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
