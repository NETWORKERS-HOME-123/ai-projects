import json
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.utils import check_random_state


def _stable_assign(ids: np.ndarray, frac_treatment: float, seed: int) -> np.ndarray:
    # Hash-like stable assignment using numpy RNG seeded by seed and id values
    # Deterministic: maps each id to a uniform number then threshold
    rs = check_random_state(seed)
    # Create per-id pseudo-random by combining seed and id in a reproducible way
    # Use a simple linear congruential step to avoid using other libs
    a = 1664525
    c = 1013904223
    mod = 2 ** 32
    vals = (a * (ids.astype(np.uint64) + seed) + c) % mod
    uniforms = (vals / mod).astype(float)
    return (uniforms < frac_treatment).astype(int)


def _diff_in_means(y, z):
    # y: outcomes, z: treatment indicator (0/1)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=int)
    n1 = z.sum()
    n0 = len(z) - n1
    if n0 == 0 or n1 == 0:
        return np.nan, np.nan
    mean1 = y[z == 1].mean()
    mean0 = y[z == 0].mean()
    var1 = y[z == 1].var(ddof=1)
    var0 = y[z == 0].var(ddof=1)
    diff = mean1 - mean0
    se = np.sqrt(var1 / n1 + var0 / n0)
    return float(diff), float(se)


def run(seed: int = 42) -> Dict[str, Any]:
    """Run the A/B testing pipeline and return a JSON-serialisable summary.

    Steps:
    - Generate synthetic experimental units and outcomes.
    - Implement stable assignment to control/treatment using _stable_assign.
    - Log exposures and compute difference-in-means per simulation.
    - Run repeated A/A simulations and compute CI coverage and sample ratio checks.
    """
    rs = check_random_state(seed)
    # Experiment parameters
    n_units = 2000
    frac_treatment = 0.5
    outcome_window_days = 7  # informational only for manifest
    sims = 200  # number of A/A repetitions for coverage evaluation
    z_alpha = 1.96  # 95% CI normal approx

    # Generate unit ids and covariates
    ids = np.arange(1, n_units + 1, dtype=np.int64)
    cov = rs.normal(size=(n_units, 3))
    # Baseline probability via logistic of a linear predictor
    lin = cov[:, 0] * 0.3 + cov[:, 1] * (-0.2) + cov[:, 2] * 0.1
    p_base = 1 / (1 + np.exp(-lin))

    # Storage for simulation results
    diffs = []
    ses = []
    sample_ratio_flags = []

    for i in range(sims):
        # For A/A we set true_treatment_effect = 0 for all units
        # Use a different seed per simulation to randomise noise but keep assignment stable per id across sims
        sim_noise = rs.normal(scale=0.1, size=n_units)
        # Observed outcomes are Bernoulli around p_base (no treatment effect)
        prob = p_base + sim_noise * 0.0
        prob = np.clip(prob, 0.001, 0.999)
        outcomes = rs.binomial(1, prob)

        # Stable assignment: depends only on id and the global seed to be auditable
        z = _stable_assign(ids, frac_treatment=frac_treatment, seed=seed)

        # Log exposures: count per arm
        n_treat = int(z.sum())
        n_ctrl = int(len(z) - n_treat)
        # Check sample ratio mismatch (>5% relative deviation flagged)
        prop_treat = n_treat / (n_treat + n_ctrl)
        flag = abs(prop_treat - frac_treatment) > 0.05
        sample_ratio_flags.append(bool(flag))

        diff, se = _diff_in_means(outcomes, z)
        diffs.append(diff)
        ses.append(se)

    diffs = np.array(diffs)
    ses = np.array(ses)

    # Compute CI coverage: whether 0 is inside diff +/- z_alpha * se
    cis_lower = diffs - z_alpha * ses
    cis_upper = diffs + z_alpha * ses
    coverage_flags = (cis_lower <= 0) & (cis_upper >= 0)
    ci_coverage = float(np.mean(coverage_flags))

    # Sample ratio mismatch rate
    sample_ratio_mismatch_rate = float(np.mean(sample_ratio_flags))

    # Summarise observed mean difference across sims
    mean_diff = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1))

    # Prepare a small auditable results table (first 5 rows) as lists
    # For auditability we include id, assigned arm and an example outcome vector from the first simulation
    example_outcomes = rs.binomial(1, p_base)
    assigned = _stable_assign(ids, frac_treatment=frac_treatment, seed=seed)
    audit_table = [
        {"id": int(int(ids[j])), "arm": int(assigned[j]), "example_outcome": int(example_outcomes[j])}
        for j in range(5)
    ]

    manifest = {
        "experiment_name": "A/A coverage check",
        "n_units": int(n_units),
        "frac_treatment": float(frac_treatment),
        "outcome_window_days": int(outcome_window_days),
        "assignment_seed": int(seed),
        "sims": int(sims),
        "stopping_rule": "fixed_horizon: run all sims; no early stopping",
    }

    results = {
        "ci_coverage": ci_coverage,
        "sample_ratio_mismatch_rate": sample_ratio_mismatch_rate,
        "mean_diff": mean_diff,
        "sd_diff": sd_diff,
        "manifest": manifest,
        "audit_table_subset": audit_table,
    }
    return results


if __name__ == "__main__":
    import json

    print(json.dumps(run(), indent=2))
