import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # required keys
    keys = [
        "silhouette_average",
        "silhouette_complete",
        "silhouette_ward",
        "dendrogram_linkage_ward",
        "representative_indices_ward",
        "stability_ward_mean_ari",
        "n_samples",
    ]
    for k in keys:
        assert k in out


def test_metric_ranges():
    out = run(42)
    # Silhouette ranges
    for k in ["silhouette_average", "silhouette_complete", "silhouette_ward"]:
        v = out[k]
        assert (v != v) or (-1.0 <= v <= 1.0)  # allow NaN in pathological case
    # stability in [0,1]
    s = out["stability_ward_mean_ari"]
    assert 0.0 <= s <= 1.0


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    # determinism for main metric
    assert a["silhouette_ward"] == b["silhouette_ward"]
    # metric floor (conservative): silhouette_ward >= 0.5
    assert a["silhouette_ward"] >= 0.5
