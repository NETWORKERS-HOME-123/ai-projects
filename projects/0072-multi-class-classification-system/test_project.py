import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    # Required keys
    assert "macro_f1" in out
    assert "confusion_matrix" in out
    assert "label_mapping" in out
    assert "example_review" in out
    # Types
    assert isinstance(out["macro_f1"], float)
    assert isinstance(out["confusion_matrix"], list)
    assert isinstance(out["label_mapping"], dict)
    assert isinstance(out["example_review"], dict)


def test_metric_ranges_and_floor():
    out = run(42)
    macro = out["macro_f1"]
    # Macro F1 must be a sensible value between 0 and 1
    assert 0.0 <= macro <= 1.0
    # Conservative metric floor for this baseline (should be comfortably met)
    assert macro >= 0.60

    # Per-class recall keys exist and are in [0,1]
    for k, v in out.items():
        if k.startswith("per_class_recall_"):
            assert 0.0 <= v <= 1.0


def test_determinism_main_metric():
    a = run(42)
    b = run(42)
    # Determinism: macro_f1 should be identical
    assert a["macro_f1"] == b["macro_f1"]
