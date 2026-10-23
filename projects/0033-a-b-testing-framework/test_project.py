import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    # Check keys exist
    expected_keys = {"ci_coverage", "sample_ratio_mismatch_rate", "mean_diff", "sd_diff", "manifest", "audit_table_subset"}
    assert isinstance(out, dict)
    assert expected_keys.issubset(set(out.keys()))
    # Types
    assert isinstance(out["ci_coverage"], float)
    assert isinstance(out["sample_ratio_mismatch_rate"], float)
    assert isinstance(out["manifest"], dict)
    assert isinstance(out["audit_table_subset"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    ci = out["ci_coverage"]
    srm = out["sample_ratio_mismatch_rate"]
    # Ranges
    assert 0.0 <= ci <= 1.0
    assert 0.0 <= srm <= 1.0
    # Conservative floor: coverage should be reasonably high for A/A with normal approx
    assert ci >= 0.90


def test_determinism():
    a = run(42)
    b = run(42)
    # Determinism for the main metric
    assert a["ci_coverage"] == b["ci_coverage"]
    # Entire dict should be equal (JSON-serialisable equivalence)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
