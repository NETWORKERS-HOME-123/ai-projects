import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"accuracy", "macro_recall", "per_folder_recall", "n_test_messages", "n_train_messages", "sample_high_conf_misclassifications", "some_suggestions"}
    assert expected_keys.issubset(set(out.keys()))
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["macro_recall"] <= 1.0
    assert isinstance(out["per_folder_recall"], dict)


def test_metric_floor_and_ranges():
    out = run(42)
    # Conservatively require reasonable accuracy on this synthetic dataset
    assert out["accuracy"] >= 0.70
    # per-folder recalls should be between 0 and 1
    for v in out["per_folder_recall"].values():
        assert 0.0 <= v <= 1.0


def test_determinism():
    a = run(42)
    b = run(42)
    # Deterministic: main metric and per-folder recalls identical
    assert a["accuracy"] == b["accuracy"]
    assert a["per_folder_recall"] == b["per_folder_recall"]
