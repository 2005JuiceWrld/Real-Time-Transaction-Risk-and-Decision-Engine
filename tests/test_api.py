from datetime import datetime, timedelta, timezone

import fakeredis
import pytest
from fastapi.testclient import TestClient

from src.serving.api import create_app


@pytest.fixture
def client():
    redis_client = fakeredis.FakeRedis(decode_responses=True)
    app = create_app(redis_client=redis_client)
    return TestClient(app)


def _score_request(**overrides) -> dict:
    payload = {
        "event_id": "evt_test_1",
        "user_id": "user_test_1",
        "event_time": datetime.now(timezone.utc).isoformat(),
        "amount": 500.0,
        "device_id": "device_1",
        "location": "Mumbai",
    }
    payload.update(overrides)
    return payload


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["redis_connected"] is True
    assert "decision_threshold" in body


def test_score_returns_a_decision(client):
    response = client.post("/score", json=_score_request())
    assert response.status_code == 200
    body = response.json()

    assert body["event_id"] == "evt_test_1"
    assert body["decision"] in {"ALLOW", "REVIEW", "BLOCK"}
    assert 0.0 <= body["risk_score"] <= 1.0
    assert isinstance(body["reason_codes"], list)


def test_score_rejects_non_positive_amount(client):
    response = client.post("/score", json=_score_request(amount=-10.0))
    assert response.status_code == 422


def test_repeated_transactions_update_online_state(client):
    """The second call for the same user should see the first one as history."""
    t0 = datetime.now(timezone.utc)

    first = client.post("/score", json=_score_request(event_id="e1", event_time=t0.isoformat()))
    assert first.status_code == 200

    second = client.post(
        "/score",
        json=_score_request(event_id="e2", event_time=(t0 + timedelta(seconds=30)).isoformat()),
    )
    assert second.status_code == 200

    features = client.get(f"/users/user_test_1/features").json()
    assert features["user_prior_event_count"] == 2


def test_metrics_endpoint_exposes_prometheus_text(client):
    client.post("/score", json=_score_request())
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "decision_engine_decisions_total" in response.text


def test_peek_features_does_not_record_state(client):
    before = client.get("/users/fresh_user/features").json()
    after = client.get("/users/fresh_user/features").json()
    assert before["user_prior_event_count"] == after["user_prior_event_count"] == 0
