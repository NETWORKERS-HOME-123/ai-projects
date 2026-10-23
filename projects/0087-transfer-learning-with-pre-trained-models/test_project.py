import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = {"accuracy_frozen", "accuracy_finetune", "accuracy_scratch", "n_components", "params_head", "model_card"}
    assert keys.issubset(set(out.keys()))
    # types
    assert isinstance(out["accuracy_finetune"], float)
    assert isinstance(out["n_components"], int)
    assert isinstance(out["model_card"], str)


def test_metrics_reasonable_ranges():
    out = run(42)
    for k in ("accuracy_frozen", "accuracy_finetune", "accuracy_scratch"):
        v = out[k]
        assert 0.0 <= v <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # determinism on the main metric
    assert a["accuracy_finetune"] == b["accuracy_finetune"]
    # conservative floor that should be comfortably met on digits data with PCA+LR
    assert a["accuracy_finetune"] >= 0.80
