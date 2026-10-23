from project import run


def test_keys_present():
    result = run(42)
    expected = {"held_out_r2", "held_out_mae", "elasticity_proxy", "baseline_price", "recommended_revenue_price", "recommended_margin_price", "baseline_revenue", "best_revenue", "best_margin"}
    assert isinstance(result, dict)
    assert expected.issubset(result.keys())


def test_metric_ranges():
    result = run(42)
    assert -1.0 <= result["held_out_r2"] <= 1.0
    assert result["held_out_mae"] >= 0
    assert result["baseline_price"] > 0
    assert 70 <= result["recommended_revenue_price"] <= 130
    assert 70 <= result["recommended_margin_price"] <= 130
    assert result["best_revenue"] >= result["baseline_revenue"]


def test_metric_floor_and_determinism():
    a = run(42)
    b = run(42)
    assert a["held_out_r2"] == b["held_out_r2"]
    assert a["held_out_r2"] >= 0.70
