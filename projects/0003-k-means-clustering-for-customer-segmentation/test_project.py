import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = {"best_k", "silhouette", "inertia", "cluster_sizes", "centroids", "stability_ari"}
    assert set(res.keys()) == expected_keys
    assert isinstance(res["best_k"], int)
    assert isinstance(res["silhouette"], float)
    assert isinstance(res["inertia"], float)
    assert isinstance(res["cluster_sizes"], list)
    assert isinstance(res["centroids"], list)
    assert isinstance(res["stability_ari"], float)


def test_metric_ranges_and_floor():
    res = run(42)
    # silhouette is between -1 and 1; expect a modest positive score for the synthetic blobs
    sil = res["silhouette"]
    assert -1.0 <= sil <= 1.0
    assert sil >= 0.35  # conservative floor the pipeline expects
    # stability between -1 and 1
    assert -1.0 <= res["stability_ari"] <= 1.0
    # sizes sum to sample count
    assert sum(res["cluster_sizes"]) == 600


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
