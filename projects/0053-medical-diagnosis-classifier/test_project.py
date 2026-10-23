import json
from project import run
import pytest


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {
        "accuracy",
        "sensitivity",
        "specificity",
        "sensitivity_ci",
        "specificity_ci",
        "brier",
        "ece",
        "subgroup_sensitivity_diff",
        "model",
    }
    assert expected_keys.issubset(set(out.keys()))
    # types
    assert isinstance(out["accuracy"], float)
    assert isinstance(out["sensitivity"], float)
    assert isinstance(out["specificity"], float)
    assert isinstance(out["sensitivity_ci"], list) and len(out["sensitivity_ci"]) == 2
    assert isinstance(out["model"], str)


def test_metric_ranges_and_floor():
    out = run(42)
    # Basic ranges
    for k in ("accuracy", "sensitivity", "specificity", "brier", "ece"):
        assert 0.0 <= out[k] <= 1.0
    # Conservative performance floor on sensitivity (held-out)
    assert out["sensitivity"] >= 0.88


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    assert a == b
    # also ensure main metric exactly equal
    assert a["sensitivity"] == b["sensitivity"]
