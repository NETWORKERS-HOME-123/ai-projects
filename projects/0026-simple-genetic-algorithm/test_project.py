import json

essential_keys = {
    "seed",
    "best_solution",
    "best_feasible_fitness",
    "optimal_fitness",
    "fraction_of_optimal",
    "fitness_history",
}

from project import run


def test_run_keys_and_types():
    out = run(42)
    assert isinstance(out, dict)
    assert essential_keys.issubset(set(out.keys()))
    assert isinstance(out["seed"], int)
    assert isinstance(out["best_solution"], list)
    assert isinstance(out["best_feasible_fitness"], int)
    assert isinstance(out["optimal_fitness"], int)
    assert isinstance(out["fraction_of_optimal"], float)
    assert isinstance(out["fitness_history"], list)


def test_metric_ranges_and_floor():
    out = run(42)
    frac = out["fraction_of_optimal"]
    best = out["best_feasible_fitness"]
    opt = out["optimal_fitness"]
    assert 0.0 <= frac <= 1.0
    assert 0 <= best <= opt
    # Conservative floor: GA should find at least 90% of optimal on this tiny instance
    assert frac >= 0.9


def test_determinism_main_metric():
    a = run(42)["fraction_of_optimal"]
    b = run(42)["fraction_of_optimal"]
    assert a == b
