import json
from project import run


def test_run_returns_expected_keys():
    r = run(42)
    assert isinstance(r, dict)
    # expected top-level keys
    for k in [
        "accuracy",
        "precision_macro",
        "recall_macro",
        "confusion_counts",
        "confusion_row_normalized",
        "manual_check_class_0",
        "class_error_explorer",
    ]:
        assert k in r


def test_metric_ranges_and_floor():
    r = run(42)
    acc = r["accuracy"]
    assert 0.0 <= acc <= 1.0
    # Conservative floor that should be comfortably met on Iris with RandomForest
    assert acc >= 0.7
    # precision and recall macros in [0,1]
    assert 0.0 <= r["precision_macro"] <= 1.0
    assert 0.0 <= r["recall_macro"] <= 1.0


def test_confusion_matrices_and_totals():
    r = run(42)
    cm = r["confusion_counts"]
    # square matrix 3x3 for Iris
    assert isinstance(cm, list)
    assert len(cm) == 3
    for row in cm:
        assert len(row) == 3
    # totals reconcile with class_error_explorer
    total_test = sum([d["total_true"] for d in r["class_error_explorer"]])
    assert total_test == sum(sum(row) for row in cm)


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # Determinism: the accuracy and confusion matrix should be identical
    assert a == b
