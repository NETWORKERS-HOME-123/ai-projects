import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # expected keys
    for k in ("vocab_size", "embedding_dim", "neighbour_quality", "main_metric", "explorer", "provenance"):
        assert k in out
    assert isinstance(out["vocab_size"], int)
    assert isinstance(out["embedding_dim"], int)
    assert isinstance(out["main_metric"], float)


def test_metric_ranges_and_floor():
    out = run(42)
    main = out["main_metric"]
    # basic valid range
    assert 0.0 <= main <= 1.0
    # conservative floor that this small pipeline comfortably passes
    assert main >= 0.3


def test_determinism():
    a = run(42)
    b = run(42)
    # main metric deterministic
    assert a["main_metric"] == b["main_metric"]
