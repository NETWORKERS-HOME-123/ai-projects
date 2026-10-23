import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    keys = [
        "median_pairwise_distance",
        "best_silhouette",
        "best_eps",
        "best_min_samples",
        "best_n_clusters",
        "best_noise_fraction",
        "explorer",
    ]
    for k in keys:
        assert k in out

    assert isinstance(out["median_pairwise_distance"], float)
    assert isinstance(out["best_eps"], float)
    assert isinstance(out["best_min_samples"], int)
    assert isinstance(out["best_n_clusters"], int)
    assert isinstance(out["best_noise_fraction"], float)
    assert isinstance(out["explorer"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    # median distance should be positive
    assert out["median_pairwise_distance"] > 0.0

    # best_noise_fraction is between 0 and 1
    assert 0.0 <= out["best_noise_fraction"] <= 1.0

    # main metric floor: best_silhouette should be at least 0.35 when defined
    sil = out["best_silhouette"]
    if sil is not None:
        assert sil >= 0.35


def test_determinism_of_main_metric():
    a = run(42)
    b = run(42)
    # Compare the main returned items deterministically
    keys = ["best_silhouette", "best_eps", "best_min_samples", "best_n_clusters", "best_noise_fraction"]
    for k in keys:
        assert a[k] == b[k]
