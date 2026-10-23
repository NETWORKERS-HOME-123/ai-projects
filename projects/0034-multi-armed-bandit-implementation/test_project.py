import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    assert "policies" in out
    assert "stationary" in out and "nonstationary" in out
    assert "epsilon_greedy" in out["stationary"]


def test_metrics_ranges_and_floor():
    out = run(42)
    eg = out["stationary"]["epsilon_greedy"]
    ucb = out["stationary"]["ucb1"]
    assert 0.0 <= eg["mean_fraction_optimal"] <= 1.0
    assert 0.0 <= ucb["mean_fraction_optimal"] <= 1.0
    assert eg["mean_total_regret"] >= 0.0
    assert ucb["mean_total_regret"] >= 0.0
    assert out["mean_fraction_optimal_epsilon_greedy"] >= 0.6


def test_determinism_of_main_metric():
    a = run(42)["mean_fraction_optimal_epsilon_greedy"]
    b = run(42)["mean_fraction_optimal_epsilon_greedy"]
    assert a == b
