import json
from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    expected_keys = {
        "content_index",
        "style_index",
        "content_image_mean",
        "style_image_mean",
        "content_mse",
        "content_corr",
        "relative_change",
        "alpha",
    }
    assert set(out.keys()) == expected_keys
    # types
    assert isinstance(out["content_index"], int)
    assert isinstance(out["style_index"], int)
    for k in ("content_image_mean", "style_image_mean", "content_mse", "content_corr", "relative_change", "alpha"):
        assert isinstance(out[k], float)


def test_metric_ranges_and_floor():
    out = run(42)
    # basic ranges
    assert out["content_mse"] >= 0.0
    assert -1.0 <= out["content_corr"] <= 1.0
    assert out["alpha"] >= 0.0 and out["alpha"] <= 1.0
    # conservative floor: correlation should be reasonably high because we blend heavily to preserve content
    assert out["content_corr"] >= 0.90


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
