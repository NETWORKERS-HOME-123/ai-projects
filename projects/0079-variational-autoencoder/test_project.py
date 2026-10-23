import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    # expected keys
    for k in ["explained_variance", "reconstruction_mse", "kl_mean", "elbo", "latent_dim"]:
        assert k in out
    assert isinstance(out["latent_dim"], int)


def test_metric_ranges_and_floor():
    out = run(42)
    # sensible numeric ranges
    assert 0.0 <= out["explained_variance"] <= 1.0
    assert out["reconstruction_mse"] >= 0.0
    assert out["kl_mean"] >= 0.0
    # conservative floor: explained variance on held-out should be at least 0.2
    assert out["explained_variance"] >= 0.2


def test_determinism():
    a = run(42)
    b = run(42)
    # Exact equality for main reported metric
    assert a["explained_variance"] == b["explained_variance"]
