import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = set(res.keys())
    expected = {"accuracy", "mean_recall", "per_class_recall", "top_confused_pairs", "example_mistakes", "viewer_sample", "n_train", "n_test"}
    assert expected.issubset(keys)
    assert 0.0 <= res["accuracy"] <= 1.0
    assert 0.0 <= res["mean_recall"] <= 1.0
    assert isinstance(res["per_class_recall"], dict)
    assert isinstance(res["top_confused_pairs"], list)
    assert isinstance(res["viewer_sample"], dict)


def test_metric_floor_and_ranges():
    res = run(42)
    # Conservative floor: accuracy should be at least 0.95 on this dataset with this model
    assert res["accuracy"] >= 0.95
    # per-class recall values within [0,1]
    for v in res["per_class_recall"].values():
        assert 0.0 <= v <= 1.0


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # determinism for accuracy and mean_recall
    assert a["accuracy"] == b["accuracy"]
    assert a["mean_recall"] == b["mean_recall"]
