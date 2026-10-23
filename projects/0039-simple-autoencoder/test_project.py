import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = {"dataset", "n_train", "n_test", "latent_dim", "test_mse_scaled", "pca_mse_scaled", "reconstruction_r2", "top_errors"}
    assert keys.issubset(set(res.keys()))
    assert res["dataset"] == "digits"
    assert isinstance(res["n_train"], int) and res["n_train"] > 0
    assert isinstance(res["n_test"], int) and res["n_test"] > 0
    assert isinstance(res["latent_dim"], int) and res["latent_dim"] > 0


def test_metric_ranges_and_floor():
    res = run(42)
    # sensible ranges
    mse = res["test_mse_scaled"]
    pca_mse = res["pca_mse_scaled"]
    r2 = res["reconstruction_r2"]
    assert 0.0 <= mse < 10.0
    assert 0.0 <= pca_mse < 10.0
    # R2 can be negative in pathological cases, but we assert a conservative floor that should hold
    assert r2 >= 0.3


def test_determinism():
    a = run(42)
    b = run(42)
    # Compare main metric exactly
    assert a["reconstruction_r2"] == b["reconstruction_r2"]
