import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = [
        "n_samples",
        "n_features",
        "missing_cells",
        "pearson_mean_abs",
        "spearman_mean_abs",
        "bootstrap_pearson_mean_abs",
        "pearson_top_abs",
    ]
    for k in keys:
        assert k in out
    assert isinstance(out["n_samples"], int)
    assert isinstance(out["n_features"], int)


def test_metric_ranges_and_floor():
    out = run(42)
    # basic ranges
    assert 0.0 <= out["pearson_mean_abs"] <= 1.0
    assert 0.0 <= out["spearman_mean_abs"] <= 1.0
    assert 0.0 <= out["pearson_top_abs"] <= 1.0
    # conservative metric floor: mean absolute Pearson should be reasonably above zero
    assert out["pearson_mean_abs"] >= 0.10


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # determinism check on a main metric
    assert a["pearson_mean_abs"] == b["pearson_mean_abs"]
