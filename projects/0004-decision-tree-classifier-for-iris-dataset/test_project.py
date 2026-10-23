import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # expected keys
    expected = {"accuracy", "accuracy_shallow", "precision_macro", "recall_macro", "confusion_matrix", "explanation"}
    assert expected.issubset(set(out.keys()))
    # per-class keys for three classes (0,1,2)
    for k in ["precision_class_0", "precision_class_1", "precision_class_2", "recall_class_0", "recall_class_1", "recall_class_2"]:
        assert k in out


def test_metric_ranges_and_floor():
    out = run(42)
    # basic ranges
    assert 0.0 <= out["accuracy"] <= 1.0
    assert 0.0 <= out["precision_macro"] <= 1.0
    assert 0.0 <= out["recall_macro"] <= 1.0
    # confusion matrix shape check
    cm = out["confusion_matrix"]
    assert isinstance(cm, list)
    assert len(cm) == 3
    # conservative accuracy floor
    assert out["accuracy"] >= 0.85


def test_determinism():
    a = run(42)["accuracy"]
    b = run(42)["accuracy"]
    assert a == b
