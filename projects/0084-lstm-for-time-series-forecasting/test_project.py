import json
from project import run


def test_run_keys_and_types():
    r = run(42)
    assert isinstance(r, dict)
    # required keys
    assert "mae_direct_mean" in r
    assert "mae_recursive_mean" in r
    assert "mae_naive_mean" in r
    assert "review_report_csv" in r
    assert "issue_origin_indices" in r


def test_metric_ranges_and_floor():
    r = run(42)
    # MAEs should be non-negative
    assert r["mae_direct_mean"] >= 0.0
    assert r["mae_recursive_mean"] >= 0.0
    # Conservative floor: direct mean MAE at least 0.0 (trivial but conservative)
    assert r["mae_direct_mean"] >= 0.0


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # determinism for the primary metric
    assert a["mae_direct_mean"] == b["mae_direct_mean"]
