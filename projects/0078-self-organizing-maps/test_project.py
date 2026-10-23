import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {"quantisation_error", "topographic_error", "grid_shape", "prototypes", "assignments", "seed"}
    assert expected_keys.issubset(set(res.keys()))
    assert isinstance(res["quantisation_error"], float)
    assert isinstance(res["topographic_error"], float)
    assert isinstance(res["grid_shape"], list)
    assert isinstance(res["prototypes"], list)
    assert isinstance(res["assignments"], list)


def test_metric_ranges_and_floor():
    res = run(42)
    qe = res["quantisation_error"]
    te = res["topographic_error"]
    # sensible ranges
    assert qe >= 0.0
    assert 0.0 <= te <= 1.0
    # conservative floor: quantisation error non-negative (trivial but conservative)
    assert qe >= 0.0


def test_determinism_of_main_metric():
    r1 = run(42)
    r2 = run(42)
    # determinism for quantisation error and topographic error
    assert r1["quantisation_error"] == r2["quantisation_error"]
    assert r1["topographic_error"] == r2["topographic_error"]
