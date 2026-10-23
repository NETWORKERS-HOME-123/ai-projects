import pytest

from project import run


def test_run_keys_and_types():
    res = run(42)
    assert isinstance(res, dict)
    keys = [
        "gru_test_accuracy",
        "gru_test_loss",
        "rnn_test_accuracy",
        "mlp_test_accuracy",
        "gru_longseq_test_accuracy",
        "runtime_seconds",
        "sequence_length_short",
        "sequence_length_long",
        "demo",
    ]
    for k in keys:
        assert k in res


def test_metric_ranges_and_floor():
    res = run(42)
    for k in ["gru_test_accuracy", "rnn_test_accuracy", "mlp_test_accuracy", "gru_longseq_test_accuracy"]:
        v = res[k]
        assert 0.0 <= v <= 1.0
    assert res["gru_test_loss"] >= 0.0
    assert res["gru_test_accuracy"] >= 0.80


def test_determinism_main_metric():
    a = run(42)["gru_test_accuracy"]
    b = run(42)["gru_test_accuracy"]
    assert a == b
