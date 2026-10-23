import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    # expected keys
    keys = {
        "final_accuracy_clean",
        "best_accuracy_clean",
        "best_round_clean",
        "final_accuracy_noisy",
        "best_accuracy_noisy",
        "best_round_noisy",
        "weak_weights",
        "noise_rate",
        "n_estimators",
    }
    assert set(out.keys()) >= keys

    # types
    assert isinstance(out["best_accuracy_clean"], float)
    assert isinstance(out["best_round_clean"], int)
    assert isinstance(out["weak_weights"], list)


def test_value_ranges_and_floor():
    out = run(42)
    # All accuracies between 0 and 1
    for k in ["final_accuracy_clean", "best_accuracy_clean", "final_accuracy_noisy", "best_accuracy_noisy"]:
        assert 0.0 <= out[k] <= 1.0

    # weak weights non-negative and length equals n_estimators
    n_est = out["n_estimators"]
    assert len(out["weak_weights"]) == n_est
    assert all(w >= 0.0 for w in out["weak_weights"]) 

    # Conservative metric floor for clean best accuracy
    assert out["best_accuracy_clean"] >= 0.80


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
