"""Online feature store tests, run against fakeredis (no real Redis needed).

The most important test here is parity: replaying the same event stream
through the online client (read-then-write, one event at a time, exactly as
a live system would) must produce the same feature values the offline batch
pipeline computes from the whole stream at once. If these ever disagree,
a model trained offline would not behave the way it does in production —
the single most common bug class in real-time ML systems.
"""

from datetime import datetime, timedelta

import fakeredis
import pandas as pd
import pytest

from src.features.feature_store_client import FeatureStoreClient
from src.features.temporal_features import FEATURE_COLUMNS, compute_features


@pytest.fixture
def client():
    r = fakeredis.FakeRedis(decode_responses=True)
    return FeatureStoreClient(r)


def test_first_event_for_a_user_has_no_history(client):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    feats = client.get_features("u1", t0, amount=100.0, device_id="dev_a", location="Mumbai")

    assert feats["is_new_device"] == 1
    assert feats["txn_count_5m"] == 0
    assert feats["user_prior_event_count"] == 0
    assert feats["amount_zscore_30d"] == 0.0


def test_get_features_does_not_mutate_state(client):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    client.get_features("u1", t0, 100.0, "dev_a", "Mumbai")
    client.get_features("u1", t0, 100.0, "dev_a", "Mumbai")

    feats = client.get_features("u1", t0, 100.0, "dev_a", "Mumbai")
    assert feats["user_prior_event_count"] == 0  # still nothing recorded


def test_record_then_read_reflects_the_recorded_event(client):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    client.record_event("u1", t0, "transaction", 100.0, "dev_a", "Mumbai")

    t1 = t0 + timedelta(minutes=2)
    feats = client.get_features("u1", t1, amount=150.0, device_id="dev_a", location="Mumbai")

    assert feats["is_new_device"] == 0  # dev_a already recorded
    assert feats["txn_count_5m"] == 1
    assert feats["user_prior_event_count"] == 1
    assert feats["seconds_since_last_event"] == pytest.approx(120.0, abs=1.0)


def test_new_device_is_flagged_until_seen(client):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    client.record_event("u1", t0, "transaction", 100.0, "dev_a", "Mumbai")

    feats_new = client.get_features("u1", t0 + timedelta(minutes=1), 100.0, "dev_b", "Mumbai")
    assert feats_new["is_new_device"] == 1

    client.record_event("u1", t0 + timedelta(minutes=1), "transaction", 100.0, "dev_b", "Mumbai")
    feats_known = client.get_features("u1", t0 + timedelta(minutes=2), 100.0, "dev_b", "Mumbai")
    assert feats_known["is_new_device"] == 0


def test_auth_failures_are_windowed(client):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    client.record_event("u1", t0, "authentication_failure", None, "dev_a", "Mumbai")
    client.record_event("u1", t0 + timedelta(minutes=10), "authentication_failure", None, "dev_a", "Mumbai")

    within_hour = client.get_features("u1", t0 + timedelta(minutes=30), 100.0, "dev_a", "Mumbai")
    assert within_hour["auth_failure_count_1h"] == 2

    next_day = client.get_features("u1", t0 + timedelta(days=2), 100.0, "dev_a", "Mumbai")
    assert next_day["auth_failure_count_1h"] == 0
    assert next_day["auth_failure_count_24h"] == 0


def test_two_users_do_not_share_state(client):
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    client.record_event("u1", t0, "transaction", 100.0, "shared_device_name", "Mumbai")

    feats_u2 = client.get_features("u2", t0 + timedelta(minutes=1), 100.0, "shared_device_name", "Mumbai")
    assert feats_u2["is_new_device"] == 1
    assert feats_u2["user_prior_event_count"] == 0


def test_online_matches_offline_feature_computation(client):
    """Replay a mixed event stream through the online client, one event at a
    time, and check every transaction's feature vector matches what the
    offline batch pipeline computes from the full stream at once."""

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    events = [
        dict(event_id="e0", user_id="u1", event_time=t0, event_type="login", amount=None,
             device_id="dev_a", location="Mumbai"),
        dict(event_id="e1", user_id="u1", event_time=t0 + timedelta(minutes=1), event_type="transaction",
             amount=100.0, device_id="dev_a", location="Mumbai"),
        dict(event_id="e2", user_id="u1", event_time=t0 + timedelta(minutes=2), event_type="transaction",
             amount=120.0, device_id="dev_a", location="Mumbai"),
        dict(event_id="e3", user_id="u1", event_time=t0 + timedelta(minutes=3), event_type="transaction",
             amount=5000.0, device_id="dev_b", location="Delhi"),
        dict(event_id="e4", user_id="u1", event_time=t0 + timedelta(hours=2), event_type="authentication_failure",
             amount=None, device_id="dev_b", location="Delhi"),
        dict(event_id="e5", user_id="u1", event_time=t0 + timedelta(hours=2, minutes=5), event_type="transaction",
             amount=200.0, device_id="dev_b", location="Delhi"),
        dict(event_id="e6", user_id="u1", event_time=t0 + timedelta(days=1), event_type="transaction",
             amount=110.0, device_id="dev_a", location="Mumbai"),
    ]

    df = pd.DataFrame(events)
    offline = compute_features(df).set_index("event_id")

    online_rows = {}
    for e in events:
        if e["event_type"] == "transaction":
            online_rows[e["event_id"]] = client.get_features(
                e["user_id"], e["event_time"], e["amount"], e["device_id"], e["location"]
            )
        client.record_event(
            e["user_id"], e["event_time"], e["event_type"], e["amount"], e["device_id"], e["location"]
        )

    assert set(online_rows) == set(offline.index)
    for event_id, online_feats in online_rows.items():
        offline_row = offline.loc[event_id]
        for col in FEATURE_COLUMNS:
            assert online_feats[col] == pytest.approx(float(offline_row[col]), abs=1e-6), (
                f"{col} mismatch for {event_id}: online={online_feats[col]} offline={offline_row[col]}"
            )
