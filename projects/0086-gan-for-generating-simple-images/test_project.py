import json
from project import run

def test_run_keys_and_types():
    metrics = run(42)
    assert isinstance(metrics, dict)
    for k in ("discriminator_accuracy", "generator_diversity", "mean_nearest_train_dist"):
        assert k in metrics
        assert isinstance(metrics[k], float)

def test_metric_ranges_and_floor():
    metrics = run(42)
    acc = metrics["discriminator_accuracy"]
    # accuracy must be between 0 and 1
    assert 0.0 <= acc <= 1.0
    # Conservative floor that should be comfortably reached on this toy problem
    assert acc >= 0.55

def test_determinism_main_metric():
    m1 = run(42)["discriminator_accuracy"]
    m2 = run(42)["discriminator_accuracy"]
    assert m1 == m2
