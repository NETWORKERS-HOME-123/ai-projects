import json
from typing import Dict, Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def generate_synthetic_transactions(seed: int = 42, n_tx: int = 2000, n_items: int = 40):
    rng = np.random.RandomState(seed)
    start = np.datetime64('2021-01-01')
    days = 90
    dates = start + rng.randint(0, days, size=n_tx).astype('timedelta64[D]')
    # base item popularity
    item_pop = rng.dirichlet(alpha=np.ones(n_items))
    items = [f'item_{i}' for i in range(n_items)]
    rows = []
    # choose 3 promoted items (their ids) to boost in certain weeks
    promoted = rng.choice(items, size=3, replace=False)
    for tx_id in range(n_tx):
        date = dates[tx_id]
        weekday = int((date - start).astype(int) % 7)
        # promotion weeks: weeks 3 and 7 (deterministic choice)
        week = int(((date - start).astype(int)) // 7)
        promo_active = week in (3, 7)
        # number of items in basket (1 to 6)
        k = rng.choice(np.arange(1, 7), p=[0.25, 0.3, 0.2, 0.13, 0.08, 0.04])
        # base sample according to popularity, boost promoted items when promo_active
        probs = item_pop.copy()
        if promo_active:
            for p in promoted:
                probs[items.index(p)] *= 3.0
        probs = probs / probs.sum()
        basket = list(rng.choice(items, size=k, replace=False, p=probs))
        # returns: small fraction marked as returns, handled by a flag
        is_return = rng.rand() < 0.03
        rows.append({'tx_id': f'TX{tx_id}', 'date': np.datetime64(date), 'items': basket, 'is_return': bool(is_return), 'promo_week': promo_active})
    return pd.DataFrame(rows)


def pair_support_confidence_lift(transactions: pd.Series, min_support: float = 0.01) -> pd.DataFrame:
    # transactions: Series of lists of items
    n = len(transactions)
    # count single item supports
    from collections import Counter
    single = Counter()
    pair = Counter()
    for items in transactions:
        unique = sorted(set(items))
        single.update(unique)
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair[(unique[i], unique[j])] += 1
    rows = []
    for (a, b), cnt in pair.items():
        sup = cnt / n
        if sup < min_support:
            continue
        sup_a = single[a] / n
        sup_b = single[b] / n
        conf_a_b = cnt / (single[a]) if single[a] > 0 else 0.0
        conf_b_a = cnt / (single[b]) if single[b] > 0 else 0.0
        # lift = conf(A->B) / support(B)
        lift_a_b = conf_a_b / sup_b if sup_b > 0 else 0.0
        lift_b_a = conf_b_a / sup_a if sup_a > 0 else 0.0
        rows.append({'antecedent': a, 'consequent': b, 'support': sup, 'count': int(cnt), 'conf_ab': conf_a_b, 'lift_ab': lift_a_b})
        rows.append({'antecedent': b, 'consequent': a, 'support': sup, 'count': int(cnt), 'conf_ab': conf_b_a, 'lift_ab': lift_b_a})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df.sort_values('lift_ab', ascending=False).reset_index(drop=True)
    return df


def evaluate_rules(train_rules: pd.DataFrame, held_transactions: pd.Series) -> Tuple[float, List[Dict]]:
    # For each rule antecedent->consequent compute held-out support and confidence
    n = len(held_transactions)
    from collections import Counter
    single = Counter()
    pair = Counter()
    for items in held_transactions:
        unique = sorted(set(items))
        single.update(unique)
        for i in range(len(unique)):
            for j in range(i + 1, len(unique)):
                pair[(unique[i], unique[j])] += 1
    eval_rows = []
    agree_count = 0
    consider = min(20, len(train_rules))
    top_rules = train_rules.head(consider)
    for _, r in top_rules.iterrows():
        a = r['antecedent']
        b = r['consequent']
        ordered_pair = tuple(sorted((a, b)))
        cnt = pair.get(ordered_pair, 0)
        sup = cnt / n if n > 0 else 0.0
        # measure agreement: held-out support > 0 (rule seen) is considered agreement
        if sup > 0.0:
            agree_count += 1
        eval_rows.append({'antecedent': a, 'consequent': b, 'held_support': sup, 'held_count': int(cnt)})
    fraction = agree_count / consider if consider > 0 else 0.0
    return fraction, eval_rows


def run(seed: int = 42) -> Dict:
    # produce a reproducible report
    df = generate_synthetic_transactions(seed=seed, n_tx=2000, n_items=40)
    # drop returns
    df = df[~df['is_return']].reset_index(drop=True)
    # split by date: use first 70% days for training, remaining for held-out
    min_date = df['date'].min()
    max_date = df['date'].max()
    # compute number of days between min and max properly
    total_days = int((max_date - min_date) / np.timedelta64(1, 'D'))
    cutoff = min_date + np.timedelta64(int(total_days * 70 // 100), 'D')
    train = df[df['date'] <= cutoff]
    held = df[df['date'] > cutoff]
    train_tx = train['items']
    held_tx = held['items']
    # mine pair rules on train
    rules = pair_support_confidence_lift(train_tx, min_support=0.01)
    n_rules = len(rules)
    avg_lift = float(rules['lift_ab'].mean()) if n_rules > 0 else 0.0
    # evaluate top rules on held-out
    held_fraction, held_details = evaluate_rules(rules, held_tx)
    # compare promoted vs non-promoted weeks: average support for promoted items vs others
    promoted_items = sorted(set(item for lst in df[df['promo_week']]['items'] for item in lst))
    all_items = sorted({it for lst in df['items'] for it in lst})
    non_promoted = [it for it in all_items if it not in promoted_items]
    def avg_support(items_list, transactions):
        if not items_list:
            return 0.0
        n = len(transactions)
        cnt = 0
        for t in transactions:
            if set(t) & set(items_list):
                cnt += 1
        return cnt / n if n > 0 else 0.0
    promo_support = avg_support(promoted_items, train_tx)
    nonpromo_support = avg_support(non_promoted, train_tx)
    report = {
        'n_transactions': int(len(df)),
        'n_items': int(len(all_items)),
        'n_rules': int(n_rules),
        'avg_train_lift': float(avg_lift),
        'promo_support': float(promo_support),
        'nonpromo_support': float(nonpromo_support),
        # main held-out metric
        'held_out_agreement': float(held_fraction)
    }
    return report


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
