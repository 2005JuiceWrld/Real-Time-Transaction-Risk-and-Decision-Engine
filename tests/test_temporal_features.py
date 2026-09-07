"""Tests for causal feature engineering.

The single most important property under test here is causality: a feature
computed for event N must never change depending on what happens after
event N. ``test_features_are_causal_wrt_future_events`` checks this directly
by comparing features computed with and without the tail of the stream.
"""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.features.temporal_features import FEATURE_COLUMNS, compute_features


def _event(
    event_id: str,
    user_id: str,
    event_time: datetime,
    event_type: str = "transaction",
    amount: float | None = 100.0,
    device_id: str = "dev_1",
    location: str = "Mumbai",
) -> dict:
    return {
        "event_id": event_id,
        "user_id": user_id,
        "event_time": event_time,
        "event_type": event_type,
        "amount": amount,
        "device_id": device_id,
        "location": location,
        "is_fraud_episode": False,
        "episode_type": None,
    }


def test_first_transaction_has_no_history():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame([_event("e1", "u1", t0)])

    feats = compute_features(df)

    assert len(feats) == 1
    row = feats.iloc[0]
    assert row["txn_count_5m"] == 0
    assert row["txn_count_24h"] == 0
    assert row["amount_zscore_30d"] == 0.0
    assert row["user_prior_event_count"] == 0
    # A device's very first use has, by definition, no prior sighting.
    assert row["is_new_device"] == 1


def test_txn_count_respects_window_boundaries():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            _event("e1", "u1", t0),
            _event("e2", "u1", t0 + timedelta(minutes=3)),  # inside 5m of e3
            _event("e3", "u1", t0 + timedelta(minutes=6)),  # outside 5m, inside 1h of e1/e2
        ]
    )

    feats = compute_features(df).set_index("event_id")

    assert feats.loc["e1", "txn_count_5m"] == 0
    assert feats.loc["e2", "txn_count_5m"] == 1  # only e1, 3 minutes prior
    assert feats.loc["e3", "txn_count_5m"] == 1  # e2 (3 min before) is in window; e1 (6 min before) is not
    assert feats.loc["e3", "txn_count_1h"] == 2  # both e1 and e2 are within the last hour


def test_is_new_device_tracks_first_use_per_device():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            _event("e1", "u1", t0, device_id="dev_a"),
            _event("e2", "u1", t0 + timedelta(hours=1), device_id="dev_a"),
            _event("e3", "u1", t0 + timedelta(hours=2), device_id="dev_b"),
        ]
    )

    feats = compute_features(df).set_index("event_id")

    assert feats.loc["e1", "is_new_device"] == 1  # dev_a, never seen before
    assert feats.loc["e2", "is_new_device"] == 0  # dev_a, seen at e1
    assert feats.loc["e3", "is_new_device"] == 1  # dev_b, never seen before


def test_amount_zscore_requires_minimum_history():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    # Two prior transactions is not enough for a meaningful z-score (needs >= 3).
    df = pd.DataFrame(
        [
            _event("e1", "u1", t0, amount=100.0),
            _event("e2", "u1", t0 + timedelta(hours=1), amount=100.0),
            _event("e3", "u1", t0 + timedelta(hours=2), amount=10_000.0),
        ]
    )

    feats = compute_features(df).set_index("event_id")
    assert feats.loc["e3", "amount_zscore_30d"] == 0.0


def test_auth_failures_are_counted_from_authentication_failure_events():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            _event("e1", "u1", t0, event_type="authentication_failure", amount=None),
            _event("e2", "u1", t0 + timedelta(minutes=10), event_type="authentication_failure", amount=None),
            _event("e3", "u1", t0 + timedelta(minutes=20)),  # the transaction being scored
        ]
    )

    feats = compute_features(df).set_index("event_id")
    assert feats.loc["e3", "auth_failure_count_1h"] == 2


def test_features_do_not_leak_the_label_column():
    assert "is_fraud_episode" not in FEATURE_COLUMNS
    assert "episode_type" not in FEATURE_COLUMNS


def test_features_are_causal_wrt_future_events():
    """Computing features with vs. without the tail of the stream must agree
    on every row that exists in both — nothing here should be a function of
    events that have not happened yet."""

    t0 = datetime(2026, 1, 1, 8, 0, 0)
    rows = []
    for i in range(12):
        rows.append(
            _event(
                f"e{i}",
                "u1",
                t0 + timedelta(minutes=15 * i),
                amount=100.0 + 10 * i,
                device_id=f"dev_{i % 3}",
            )
        )
    # A second user, interleaved in time, to make sure per-user grouping
    # doesn't leak across users either.
    for i in range(6):
        rows.append(_event(f"f{i}", "u2", t0 + timedelta(minutes=20 * i), amount=500.0))

    full_df = pd.DataFrame(rows)

    full_feats = compute_features(full_df).set_index("event_id")
    truncated_feats = compute_features(full_df.iloc[:8]).set_index("event_id")

    common_ids = truncated_feats.index
    pd.testing.assert_frame_equal(
        full_feats.loc[common_ids, FEATURE_COLUMNS],
        truncated_feats.loc[common_ids, FEATURE_COLUMNS],
        check_like=False,
    )


def test_multiple_users_do_not_share_state():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            _event("e1", "u1", t0, device_id="shared_looking_name"),
            _event("e2", "u2", t0 + timedelta(minutes=1), device_id="shared_looking_name"),
        ]
    )

    feats = compute_features(df).set_index("event_id")
    # Same device string, but each user is seeing it for the first time —
    # u2's row must not inherit u1's history just because the strings match.
    assert feats.loc["e1", "is_new_device"] == 1
    assert feats.loc["e2", "is_new_device"] == 1


def test_only_transaction_rows_are_returned():
    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame(
        [
            _event("e1", "u1", t0, event_type="login", amount=None),
            _event("e2", "u1", t0 + timedelta(minutes=1)),
        ]
    )

    feats = compute_features(df)
    assert list(feats["event_id"]) == ["e2"]


def test_missing_label_column_is_tolerated():
    """compute_features must also work on data with no label at all —
    the shape it will be called with at real-time scoring time."""

    t0 = datetime(2026, 1, 1, 12, 0, 0)
    df = pd.DataFrame([_event("e1", "u1", t0)]).drop(columns=["is_fraud_episode", "episode_type"])

    feats = compute_features(df)
    assert "is_fraud_episode" not in feats.columns
    assert len(feats) == 1
