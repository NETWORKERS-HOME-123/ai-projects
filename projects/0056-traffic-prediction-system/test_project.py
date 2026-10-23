import json
import math
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {
        "held_segments_r2",
        "held_segments_rmse",
        "held_segments_rush_rmse",
        "held_segments_incident_rmse",
        "future_r2",
        "future_rmse",
        "future_persistence_rmse",
        "planning",
    }
    assert expected_keys.issubset(set(out.keys()))
    assert isinstance(out["held_segments_r2"], float)
    assert isinstance(out["held_segments_rmse"], float)
    assert isinstance(out["planning"], dict)


def test_metric_ranges_and_sane_values():
    out = run(42)
    assert math.isfinite(out["held_segments_r2"]) and -1e3 < out["held_segments_r2"] < 1e3
    assert out["held_segments_rmse"] >= 0.0
    assert out["future_rmse"] >= 0.0
    assert out["future_persistence_rmse"] >= 0.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    assert a["held_segments_r2"] == b["held_segments_r2"]
    assert a["held_segments_r2"] >= 0.30
