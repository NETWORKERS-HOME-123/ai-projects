import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # expected keys
    for k in ["best_k", "n_segments", "silhouette", "stability_ari", "segment_summary"]:
        assert k in out
    assert isinstance(out["best_k"], int)
    assert isinstance(out["n_segments"], int)
    assert isinstance(out["silhouette"], float)
    assert isinstance(out["stability_ari"], float)
    assert isinstance(out["segment_summary"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    sil = out["silhouette"]
    ari = out["stability_ari"]
    # silhouette valid range approx [-1, 1]
    assert -1.0 <= sil <= 1.0
    # ARI valid range [0,1] for our synthetic aligned clusters
    assert 0.0 <= ari <= 1.0
    # Conservative metric floor that this synthetic pipeline comfortably meets
    assert ari >= 0.6


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
