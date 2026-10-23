import json
from project import run

def test_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {
        "test_mean_return",
        "random_mean_return",
        "improvement",
        "n_test_seeds",
        "replay_states",
        "replay_actions",
        "replay_rewards",
        "failure_states",
    }
    assert expected_keys.issubset(set(res.keys()))

def test_metric_ranges():
    res = run(42)
    assert isinstance(res["test_mean_return"], float)
    assert isinstance(res["random_mean_return"], float)
    assert -10.0 < res["test_mean_return"] < 10.0
    assert -10.0 < res["random_mean_return"] < 10.0
    assert isinstance(res["n_test_seeds"], int) and res["n_test_seeds"] > 0

def test_metric_floor():
    res = run(42)
    assert res["test_mean_return"] >= 0.5

def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
