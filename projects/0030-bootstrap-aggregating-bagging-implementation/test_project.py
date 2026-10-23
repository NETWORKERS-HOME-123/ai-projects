import json
from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    expected_keys = [
        "holdout_accuracy_100",
        "oob_score_100",
        "mean_fraction_included_per_tree",
        "aggregated_holdout_accuracy",
        "mean_tree_agreement_with_aggregate",
        "per_tree_preds_first_10_rows",
        "all",
    ]
    for k in expected_keys:
        assert k in res

    # types
    assert isinstance(res["holdout_accuracy_100"], float)
    assert isinstance(res["oob_score_100"], float)
    assert isinstance(res["mean_fraction_included_per_tree"], float)
    assert isinstance(res["per_tree_preds_first_10_rows"], list)


def test_metric_ranges_and_floor():
    res = run(42)
    ha = res["holdout_accuracy_100"]
    oob = res["oob_score_100"]
    agree = res["mean_tree_agreement_with_aggregate"]
    frac = res["mean_fraction_included_per_tree"]

    # basic valid ranges
    assert 0.0 <= ha <= 1.0
    assert 0.0 <= oob <= 1.0
    assert 0.0 <= agree <= 1.0
    assert 0.0 <= frac <= 1.0

    # conservative performance floor that this setup should comfortably pass
    assert ha >= 0.7


def test_determinism():
    a = run(42)
    b = run(42)
    assert a == b
