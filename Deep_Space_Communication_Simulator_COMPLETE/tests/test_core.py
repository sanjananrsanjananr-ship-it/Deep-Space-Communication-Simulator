import pytest
from src.communication_engine import link_metrics, link_quality
from src.packet_service import DTNQueue
from src.auth_utils import hash_password, verify_password

def test_link_metrics_sane():
    m = link_metrics(1000)
    assert m["distance"] == 1000
    assert m["delay_s"] > 0
    assert 0 <= m["packet_loss"] <= 1
    assert m["throughput_bps"] >= 0

def test_link_metrics_rejects_invalid_distance():
    with pytest.raises(ValueError):
        link_metrics(0)

def test_quality_labels():
    assert link_quality(15, 0.01) == "GOOD"
    assert link_quality(5, 0.10) == "MODERATE"
    assert link_quality(-2, 0.9) == "POOR"

def test_dtn_store_forward_fifo():
    q = DTNQueue()
    q.store("A", "EARTH")
    q.store("B", "MARS")
    assert len(q) == 2
    assert q.forward()["payload"] == "A"
    assert q.forward()["payload"] == "B"
    assert q.forward() is None

def test_password_hash():
    stored = hash_password("correct-horse")
    assert verify_password("correct-horse", stored)
    assert not verify_password("wrong", stored)

def test_ml_split_is_80_20():
    from src.ml_service import train_evaluate
    _, metrics = train_evaluate(samples=100)
    assert metrics["train_samples"] == 80
    assert metrics["test_samples"] == 20
