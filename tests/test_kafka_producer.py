"""Unit tests for the producer's pure logic.

Deliberately does not test ``replay``/``main`` here — those need a live
Kafka broker (``kafka.KafkaProducer`` connects on construction) and are
exercised instead by the docker-compose end-to-end path documented in the
README.
"""

import json
import time

import numpy as np
import pandas as pd

from src.producers.kafka_producer import _flush_due, _serialize


class _FakeProducer:
    def __init__(self):
        self.sent: list[tuple[str, dict]] = []

    def send(self, topic, key, value):
        self.sent.append((key, value))


def test_serialize_drops_ground_truth_label_columns():
    row = pd.Series(
        {
            "event_id": "e1",
            "user_id": "u1",
            "event_time": pd.Timestamp("2026-01-01T12:00:00"),
            "event_type": "transaction",
            "amount": 100.0,
            "is_fraud_episode": True,
            "episode_type": "account_takeover",
        }
    )

    payload = _serialize(row)

    assert "is_fraud_episode" not in payload
    assert "episode_type" not in payload
    assert payload["event_id"] == "e1"
    assert "produced_at" in payload


def test_serialize_converts_pandas_nan_to_json_null_not_the_nan_token():
    """A login/auth-failure row has no amount: pandas represents that
    missing value as float NaN, not None. json.dumps will happily emit the
    non-standard `NaN` token for a bare NaN, which a strict JSON parser
    downstream rejects, and which Pydantic's `ge=0` constraint fails outright
    (NaN compares false to everything) — every such event was being dropped
    as malformed before this was fixed."""

    row = pd.Series(
        {
            "event_id": "e1",
            "user_id": "u1",
            "event_time": pd.Timestamp("2026-01-01T12:00:00"),
            "event_type": "login",
            "amount": np.nan,
            "merchant_id": np.nan,
        }
    )

    payload = _serialize(row)

    assert payload["amount"] is None
    assert payload["merchant_id"] is None
    # Round-trips as valid, standard JSON — `NaN` is not.
    reparsed = json.loads(json.dumps(payload))
    assert reparsed["amount"] is None


def test_serialize_event_time_is_isoformat_string():
    row = pd.Series({"event_time": pd.Timestamp("2026-01-01T12:00:00"), "amount": 1.0})
    payload = _serialize(row)
    assert payload["event_time"] == "2026-01-01T12:00:00"


def test_flush_due_sends_only_expired_entries():
    producer = _FakeProducer()
    now = time.monotonic()
    pending = [
        (now - 1, {"user_id": "u1", "tag": "already due"}),
        (now + 100, {"user_id": "u2", "tag": "not due yet"}),
    ]

    remaining = _flush_due(producer, pending)

    assert len(producer.sent) == 1
    assert producer.sent[0][1]["tag"] == "already due"
    assert len(remaining) == 1
    assert remaining[0][1]["tag"] == "not due yet"


def test_flush_due_with_nothing_pending():
    producer = _FakeProducer()
    assert _flush_due(producer, []) == []
    assert producer.sent == []
