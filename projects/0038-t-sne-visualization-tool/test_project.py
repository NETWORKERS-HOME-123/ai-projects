import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {"dataset", "n_samples", "n_features", "n_classes", "numpy_version", "pandas_version", "sklearn_version", "perplexities", "seeds", "runs", "stability", "avg_trustworthiness", "warning"}
    assert expected_keys.issubset(set(out.keys()))
    assert out["dataset"] == "iris"
    assert isinstance(out["n_samples"], int) and out["n_samples"] > 0


def test_metric_ranges_and_floor():
    out = run(42)
    runs = out["runs"]
    assert isinstance(runs, list) and len(runs) >= 3
    for r in runs:
        assert 0.0 <= r["trustworthiness"] <= 1.0
        assert -1.0 <= r["distance_correlation"] <= 1.0
    # Conservative floor for average trustworthiness (this dataset/setting comfortably exceeds this)
    assert out["avg_trustworthiness"] >= 0.9


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    # Deterministic main summary metric
    assert a["avg_trustworthiness"] == b["avg_trustworthiness"]
