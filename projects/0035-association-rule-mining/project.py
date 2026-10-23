import json
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def _make_synthetic_transactions(
    n_tx: int, items: List[str], rng: np.random.Generator
) -> pd.DataFrame:
    df = pd.DataFrame(0, index=np.arange(n_tx), columns=items, dtype=int)
    # Three deterministic co-purchase patterns plus some empty baskets.
    patterns = [("item_0", "item_1"), ("item_2", "item_3"), ("item_4", "item_5")]
    probs = np.array([0.28, 0.24, 0.20, 0.28])
    choices = rng.choice(4, size=n_tx, p=probs)
    for i, c in enumerate(choices):
        if c < 3:
            a, b = patterns[c]
            df.loc[i, [a, b]] = 1
    # Rare noise items stay below the support threshold.
    df.loc[rng.random(n_tx) < 0.02, "item_6"] = 1
    df.loc[rng.random(n_tx) < 0.02, "item_7"] = 1
    return df


def _frequent_itemsets(df: pd.DataFrame, max_len: int, min_support: float) -> Dict[Tuple[str, ...], float]:
    n = len(df)
    items = list(df.columns)
    supports = {}
    for k in range(1, max_len + 1):
        for combo in itertools.combinations(items, k):
            supp = df[list(combo)].all(axis=1).sum() / n
            if supp >= min_support:
                supports[combo] = float(supp)
    return supports


def _derive_rules(supports: Dict[Tuple[str, ...], float]) -> List[Dict]:
    rules = []
    for itemset, s_itemset in supports.items():
        if len(itemset) < 2:
            continue
        itemset_set = set(itemset)
        for r in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, r):
                antecedent = tuple(sorted(antecedent))
                consequent = tuple(sorted(itemset_set - set(antecedent)))
                s_ante = supports.get(antecedent, 0.0)
                s_cons = supports.get(consequent, 0.0)
                if s_ante <= 0:
                    continue
                confidence = s_itemset / s_ante
                lift = confidence / s_cons if s_cons > 0 else float("inf")
                rules.append(
                    {
                        "antecedent": antecedent,
                        "consequent": consequent,
                        "support": float(s_itemset),
                        "confidence": float(confidence),
                        "lift": float(lift),
                    }
                )
    rules.sort(key=lambda r: (-r["lift"], -r["confidence"], r["antecedent"], r["consequent"]))
    return rules


def run(seed: int = 42) -> Dict:
    rng = np.random.default_rng(seed)
    items = [f"item_{i}" for i in range(8)]
    n_total = 1200
    n_train = int(n_total * 0.7)
    n_test = n_total - n_train

    df_train = _make_synthetic_transactions(n_train, items, rng)
    df_test = _make_synthetic_transactions(n_test, items, rng)

    min_support = 0.05
    supports_train = _frequent_itemsets(df_train, max_len=3, min_support=min_support)
    rules_train = _derive_rules(supports_train)

    n_test = len(df_test)
    rules_eval = []
    for r in rules_train:
        ant = list(r["antecedent"])
        cons = list(r["consequent"])
        s_itemset_test = df_test[ant + cons].all(axis=1).sum() / max(1, n_test)
        s_ant_test = df_test[ant].all(axis=1).sum() / max(1, n_test)
        s_cons_test = df_test[cons].all(axis=1).sum() / max(1, n_test)
        conf_test = float(s_itemset_test / s_ant_test) if s_ant_test > 0 else 0.0
        lift_test = float(conf_test / s_cons_test) if s_cons_test > 0 else float("inf")
        rules_eval.append(
            {
                "antecedent": tuple(ant),
                "consequent": tuple(cons),
                "support_test": float(s_itemset_test),
                "confidence_test": float(conf_test),
                "lift_test": float(lift_test),
            }
        )

    tol = 0.20
    stable = 0
    for r_tr, r_te in zip(rules_train, rules_eval):
        c_tr = r_tr["confidence"]
        c_te = r_te["confidence_test"]
        if c_tr > 0 and abs(c_tr - c_te) / max(c_tr, 1e-12) <= tol:
            stable += 1
    stability_fraction = stable / max(1, len(rules_train))

    top_rules = []
    for r in rules_train[:10]:
        top_rules.append(
            {
                "rule": f"{','.join(r['antecedent'])} -> {','.join(r['consequent'])}",
                "support": r["support"],
                "confidence": r["confidence"],
                "lift": r["lift"],
                "count": int(round(r["support"] * len(df_train))),
            }
        )

    return {
        "n_transactions_train": int(len(df_train)),
        "n_transactions_test": int(len(df_test)),
        "n_rules": int(len(rules_train)),
        "avg_lift": float(np.mean([r["lift"] for r in rules_train]) if rules_train else 0.0),
        "stability_fraction": float(stability_fraction),
        "top_rules": top_rules,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
