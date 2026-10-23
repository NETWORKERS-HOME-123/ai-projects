import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {
        "dataset_version",
        "accuracy",
        "macro_f1",
        "per_class",
        "confusion_matrix",
        "roc_auc_ovr",
        "exportable_error_summary",
    }
    assert expected_keys.issubset(set(out.keys()))
    # types
    assert isinstance(out["dataset_version"], str)
    assert isinstance(out["accuracy"], float)
    assert isinstance(out["macro_f1"], float)
    assert isinstance(out["per_class"], dict)
    assert isinstance(out["confusion_matrix"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    acc = out["accuracy"]
    # sanity ranges
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= out["macro_f1"] <= 1.0
    # conservative floor: accuracy should be at least 0.80 on iris with LogisticRegression
    assert acc >= 0.8


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    assert a["accuracy"] == b["accuracy"]
    # entire dict should be reproducible as well
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
