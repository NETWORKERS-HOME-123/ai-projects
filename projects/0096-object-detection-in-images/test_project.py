import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # expected keys
    keys = {"map50", "map75", "mean_iou", "class_accuracy", "viewer_sample_count", "viewer_sample"}
    assert keys.issubset(set(out.keys()))
    assert 0.0 <= out["map50"] <= 1.0
    assert 0.0 <= out["map75"] <= 1.0
    assert 0.0 <= out["mean_iou"] <= 1.0
    assert 0.0 <= out["class_accuracy"] <= 1.0


def test_metric_floor_and_ranges():
    out = run(42)
    # Conservative floor on primary held-out metric
    assert out["map50"] >= 0.60
    # mean IoU should be reasonable for boxes from same small images
    assert out["mean_iou"] >= 0.60
    # classification should be good on digits with RandomForest
    assert out["class_accuracy"] >= 0.80


def test_determinism():
    a = run(42)
    b = run(42)
    # Compare main metric
    assert a["map50"] == b["map50"]
    assert a["mean_iou"] == b["mean_iou"]
