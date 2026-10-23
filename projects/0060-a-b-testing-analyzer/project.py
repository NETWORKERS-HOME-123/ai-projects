import json
from typing import Dict
import numpy as np
import pandas as pd
from math import sqrt
import math

def _norm_cdf(x: float) -> float:
    # Standard normal CDF via erf
    return 0.5 * (1 + math.erf(x / sqrt(2)))

def simulate_experiment(n: int, p_control: float, effect: float, missing_rate: float, seed: int) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    assigned = rng.binomial(1, 0.5, size=n)  # 0 -> control, 1 -> treatment
    probs = np.where(assigned == 1, p_control + effect, p_control)
    probs = np.clip(probs, 0.0, 1.0)
    outcomes = rng.binomial(1, probs)
    # Introduce missing outcomes at random
    miss_mask = rng.rand(n) < missing_rate
    outcomes = outcomes.astype(object)
    outcomes[miss_mask] = None
    df = pd.DataFrame({
        "assigned": assigned,
        "outcome": outcomes,
    })
    return df

def two_prop_ztest(n1, x1, n2, x2):
    # returns z, p (two-sided), diff = p2 - p1
    p1 = x1 / n1
    p2 = x2 / n2
    diff = p2 - p1
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    if se == 0:
        return 0.0, 1.0, diff
    z = diff / se
    pval = 2 * (1 - _norm_cdf(abs(z)))
    return z, pval, diff

def proportion_ci(p, n, z=1.96):
    # Normal approximation CI for a single proportion
    se = sqrt(max(p * (1 - p), 0.0) / n) if n > 0 else 0.0
    return p - z * se, p + z * se

def run(seed: int = 42) -> Dict:
    # Parameters (kept simple and fixed-horizon)
    n = 2000
    p_control = 0.10
    true_effect = 0.05  # additive increase in conversion probability for treatment
    missing_rate = 0.02
    alpha = 0.05
    stopping_rule = f"Fixed-horizon: n={n}, alpha={alpha}"

    df = simulate_experiment(n=n, p_control=p_control, effect=true_effect, missing_rate=missing_rate, seed=seed)

    # Data validation: sample ratio mismatch (SRM)
    n_total = len(df)
    n_A = int((df["assigned"] == 0).sum())
    n_B = int((df["assigned"] == 1).sum())
    p_obs = n_A / n_total
    p_exp = 0.5
    se_srm = sqrt(p_exp * (1 - p_exp) / n_total)
    z_srm = (p_obs - p_exp) / se_srm
    srm_pvalue = 2 * (1 - _norm_cdf(abs(z_srm)))

    # Outcome missingness
    missing_rate_obs = df["outcome"].isna().mean()

    # Prepare counts excluding missing
    observed = df.dropna(subset=["outcome"]).copy()
    n1 = int((observed["assigned"] == 0).sum())
    n2 = int((observed["assigned"] == 1).sum())
    x1 = int(((observed["assigned"] == 0) & (observed["outcome"] == 1)).sum())
    x2 = int(((observed["assigned"] == 1) & (observed["outcome"] == 1)).sum())

    # Main test and CI
    z, p_value, abs_diff = two_prop_ztest(n1, x1, n2, x2)
    p1 = x1 / n1 if n1 > 0 else 0.0
    p2 = x2 / n2 if n2 > 0 else 0.0
    rel_diff = (p2 - p1) / p1 if p1 > 0 else float("inf")

    # Confidence interval for absolute difference via normal approx
    se = sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2) if n1 > 0 and n2 > 0 else 0.0
    z_crit = 1.96
    ci_lower = abs_diff - z_crit * se
    ci_upper = abs_diff + z_crit * se

    # A/A simulations to estimate false-positive rate (Type I error under no effect)
    sims = 200
    rng = np.random.RandomState(seed)
    aa_pvals = []
    for i in range(sims):
        # different seeds per sim for reproducibility but deterministic overall
        sim_seed = int(rng.randint(0, 2 ** 31 - 1))
        sim = simulate_experiment(n=n, p_control=p_control, effect=0.0, missing_rate=missing_rate, seed=sim_seed)
        sim_obs = sim.dropna(subset=["outcome"]).copy()
        n1s = int((sim_obs["assigned"] == 0).sum())
        n2s = int((sim_obs["assigned"] == 1).sum())
        x1s = int(((sim_obs["assigned"] == 0) & (sim_obs["outcome"] == 1)).sum())
        x2s = int(((sim_obs["assigned"] == 1) & (sim_obs["outcome"] == 1)).sum())
        _, p_sim, _ = two_prop_ztest(n1s, x1s, n2s, x2s)
        aa_pvals.append(p_sim)
    aa_pvals = np.array(aa_pvals)
    aa_false_positive_rate = float((aa_pvals < alpha).mean())

    report = {
        "n_total": int(n_total),
        "n_A": int(n_A),
        "n_B": int(n_B),
        "missing_rate": float(missing_rate_obs),
        "srm_pvalue": float(srm_pvalue),
        "abs_effect": float(abs_diff),
        "rel_effect": float(rel_diff),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value": float(p_value),
        "alpha": float(alpha),
        "stopping_rule": stopping_rule,
        "aa_false_positive_rate": float(aa_false_positive_rate),
    }
    return report

if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
