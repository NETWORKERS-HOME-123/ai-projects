import json
from project import run
import math

def test_run_structure_and_types():
    rpt = run(42)
    # expected keys
    keys = {"n_total","n_A","n_B","missing_rate","srm_pvalue","abs_effect","rel_effect","ci_lower","ci_upper","p_value","alpha","stopping_rule","aa_false_positive_rate"}
    assert set(rpt.keys()) >= keys
    # types
    assert isinstance(rpt["n_total"], int)
    assert isinstance(rpt["n_A"], int)
    assert isinstance(rpt["n_B"], int)
    assert isinstance(rpt["stopping_rule"], str)

def test_metric_ranges_and_consistency():
    rpt = run(42)
    assert 0.0 <= rpt["missing_rate"] < 0.2
    assert 0.0 <= rpt["srm_pvalue"] <= 1.0
    assert rpt["ci_lower"] <= rpt["ci_upper"]
    assert 0.0 <= rpt["p_value"] <= 1.0
    assert 0.0 <= rpt["aa_false_positive_rate"] <= 1.0

def test_metric_floor_and_determinism():
    r1 = run(42)
    r2 = run(42)
    # determinism check on main metric
    assert r1["aa_false_positive_rate"] == r2["aa_false_positive_rate"]
    # conservative floor: the A/A false-positive estimate should be numeric and at least 0.0
    assert r1["aa_false_positive_rate"] >= 0.0
