import json
from project import run


def test_run_keys():
    res = run(42)
    assert isinstance(res, dict)
    assert set(res.keys()) == {"silhouette", "inertia", "n_clusters"}


def test_metric_ranges_and_floor():
    res = run(42)
    sil = res["silhouette"]
    inertia = res["inertia"]
    n_clusters = res["n_clusters"]
    # silhouette: must be in [-1, 1] or NaN; we expect a valid score here
    assert isinstance(n_clusters, int) and n_clusters >= 2
    assert isinstance(inertia, float) and inertia >= 0.0
    assert isinstance(sil, float)
    # Floor: expect reasonable clustering quality on these synthetic images
    assert sil >= 0.25


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
