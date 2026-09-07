"""Causal, temporal feature engineering.

Every feature here is computed using *strictly prior* events only — this is
the boundary that keeps the model honest. Given the full event stream up to
and including some transaction, a feature may look at anything with
``event_time`` earlier than that transaction, plus the transaction's own
``amount``/``device_id``/``event_time`` (all genuinely known the moment a
real system would have to score it). It may never look at
``is_fraud_episode`` or ``episode_type`` — those are simulation ground
truth, not something a real system would know at scoring time.

The same function is used to build the offline training table
(``src/models/train.py``) and to explain, feature-by-feature, what the online
feature store (``src/features/feature_store_client.py``) needs to maintain —
the two are deliberately kept in lock-step so online/offline skew is a
non-issue by construction rather than by discipline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Single source of truth for the model's input columns. Training, the
# decision engine, and the FastAPI serving layer all import this list rather
# than each hard-coding their own — a mismatch here is exactly how real
# systems develop silent train/serve skew.
FEATURE_COLUMNS: list[str] = [
    "amount",
    "seconds_since_last_event",
    "seconds_since_last_txn",
    "txn_count_5m",
    "txn_count_1h",
    "txn_count_24h",
    "amount_mean_24h",
    "amount_std_24h",
    "amount_mean_30d",
    "amount_std_30d",
    "amount_zscore_30d",
    "unique_devices_24h",
    "is_new_device",
    "auth_failure_count_1h",
    "auth_failure_count_24h",
    "distinct_locations_24h",
    "user_prior_event_count",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
]

LABEL_COLUMN = "is_fraud_episode"
ID_COLUMNS = ["event_id", "user_id", "event_time"]

# Shared with src/features/feature_store_client.py so the online feature
# store keeps state for exactly the windows the offline features need.
# Kept as numpy timedelta64 (not pandas Timedelta) so subtracting one from a
# numpy datetime64 scalar stays plain numpy arithmetic for np.searchsorted.
WINDOWS = {
    "5m": np.timedelta64(5, "m"),
    "1h": np.timedelta64(1, "h"),
    "24h": np.timedelta64(24, "h"),
    "30d": np.timedelta64(30, "D"),
}
LARGE_GAP_SECONDS = 30 * 24 * 3600.0  # sentinel for "no prior event"; public, reused by feature_store_client.py
_EPS = 1e-6


def cyclical(value, period: float) -> tuple[float, float]:
    """sin/cos encoding of a periodic value (hour-of-day, day-of-week, ...).

    Public — ``src/features/feature_store_client.py`` uses this exact
    function so the online path's hour/day encoding matches the offline
    training table bit-for-bit, not just "approximately the same idea".
    """
    angle = 2 * np.pi * value / period
    return np.sin(angle), np.cos(angle)


# Back-compat alias for the array-oriented internal call sites below.
_cyclical = cyclical


def _window_slice(times: np.ndarray, hi: int, window: np.timedelta64) -> tuple[int, int]:
    """(lo, hi) bounding all entries of ``times`` in ``[times[hi] - window, times[hi])``.

    ``times`` must be sorted ascending. ``hi`` is treated as an exclusive
    upper bound — i.e. ``times[:hi]`` are the only entries considered, which
    is exactly "everything strictly before the event currently being scored".
    """
    if hi == 0:
        return 0, 0
    lo = int(np.searchsorted(times[:hi], times[hi] - window, side="left"))
    return lo, hi


def _user_features(g: pd.DataFrame) -> pd.DataFrame:
    """Compute every feature in ``FEATURE_COLUMNS`` for one user's sorted event stream.

    ``g`` must already be sorted by ``event_time`` ascending and reindexed
    0..n-1. Returns a frame aligned to ``g``'s index restricted to the
    transaction rows (decisions are only ever made on transactions).
    """

    n = len(g)
    times = g["event_time"].to_numpy(dtype="datetime64[ns]")
    event_types = g["event_type"].to_numpy()
    devices = g["device_id"].to_numpy()
    locations = g["location"].to_numpy()
    amounts = g["amount"].fillna(0.0).to_numpy(dtype=float)

    is_txn = event_types == "transaction"
    is_auth_fail = event_types == "authentication_failure"

    txn_positions = np.where(is_txn)[0]
    txn_times = times[txn_positions]
    txn_amounts = amounts[txn_positions]
    auth_fail_times = times[is_auth_fail]

    seen_devices: set[str] = set()
    device_first_seen_before: dict[int, bool] = {}
    # Single forward pass to know, for every row, whether its device had
    # already appeared in some *strictly earlier* event of any type.
    for i in range(n):
        device_first_seen_before[i] = devices[i] in seen_devices
        seen_devices.add(devices[i])

    rows = []
    for rank, i in enumerate(txn_positions):
        t = times[i]

        prior_all_lo_24h, prior_all_hi = _window_slice(times, i, WINDOWS["24h"])
        window_devices = devices[prior_all_lo_24h:prior_all_hi]
        window_locations = locations[prior_all_lo_24h:prior_all_hi]

        prior_auth_hi = int(np.searchsorted(auth_fail_times, t, side="left"))
        prior_auth_lo_1h = int(np.searchsorted(auth_fail_times[:prior_auth_hi], t - WINDOWS["1h"], side="left"))
        prior_auth_lo_24h = int(np.searchsorted(auth_fail_times[:prior_auth_hi], t - WINDOWS["24h"], side="left"))

        count_5m, _, _ = _txn_window_stats(txn_times, txn_amounts, rank, WINDOWS["5m"])
        count_1h, _, _ = _txn_window_stats(txn_times, txn_amounts, rank, WINDOWS["1h"])
        count_24h, mean_24h, std_24h = _txn_window_stats(txn_times, txn_amounts, rank, WINDOWS["24h"])
        _count_30d, mean_30d, std_30d = _txn_window_stats(txn_times, txn_amounts, rank, WINDOWS["30d"])

        seconds_since_last_event = (
            float((t - times[i - 1]) / np.timedelta64(1, "s")) if i > 0 else LARGE_GAP_SECONDS
        )
        seconds_since_last_txn = (
            float((t - txn_times[rank - 1]) / np.timedelta64(1, "s")) if rank > 0 else LARGE_GAP_SECONDS
        )

        current_amount = amounts[i]
        # A z-score from 1-2 samples is meaningless, and a std that happens to
        # land near zero blows the score up arbitrarily rather than signalling
        # anything — both are floored so this stays a real anomaly measure.
        if _count_30d >= 3:
            std_floor = max(std_30d, 0.05 * mean_30d, 1.0)
            zscore = (current_amount - mean_30d) / std_floor
        else:
            zscore = 0.0

        hour = pd.Timestamp(t).hour + pd.Timestamp(t).minute / 60.0
        dow = pd.Timestamp(t).dayofweek
        hour_sin, hour_cos = _cyclical(np.array(hour), 24.0)
        dow_sin, dow_cos = _cyclical(np.array(dow), 7.0)

        rows.append(
            {
                "_row_index": i,
                "amount": current_amount,
                "seconds_since_last_event": seconds_since_last_event,
                "seconds_since_last_txn": seconds_since_last_txn,
                "txn_count_5m": count_5m,
                "txn_count_1h": count_1h,
                "txn_count_24h": count_24h,
                "amount_mean_24h": mean_24h,
                "amount_std_24h": std_24h,
                "amount_mean_30d": mean_30d,
                "amount_std_30d": std_30d,
                "amount_zscore_30d": zscore,
                "unique_devices_24h": len(set(window_devices)),
                "is_new_device": 0 if device_first_seen_before[i] else 1,
                "auth_failure_count_1h": prior_auth_hi - prior_auth_lo_1h,
                "auth_failure_count_24h": prior_auth_hi - prior_auth_lo_24h,
                "distinct_locations_24h": len(set(window_locations)) if len(window_locations) else 1,
                "user_prior_event_count": i,
                "hour_sin": float(hour_sin),
                "hour_cos": float(hour_cos),
                "dow_sin": float(dow_sin),
                "dow_cos": float(dow_cos),
            }
        )

    out = pd.DataFrame(rows).set_index("_row_index")
    return out


def _txn_window_stats(
    txn_times: np.ndarray,
    txn_amounts: np.ndarray,
    rank: int,
    window: np.timedelta64,
) -> tuple[int, float, float]:
    """Count/mean/std of transactions strictly before position ``rank`` within ``window``.

    ``rank`` is the transaction's own position within ``txn_times`` (which
    holds only transaction-type events), so ``txn_times[:rank]`` is exactly
    "this user's prior transactions", already excluding the one being scored.
    """
    if rank == 0:
        return 0, 0.0, 0.0
    t = txn_times[rank]
    lo = int(np.searchsorted(txn_times[:rank], t - window, side="left"))
    count = rank - lo
    if count == 0:
        return 0, 0.0, 0.0
    window_amounts = txn_amounts[lo:rank]
    return count, float(window_amounts.mean()), float(window_amounts.std(ddof=0))


def compute_features(events: pd.DataFrame) -> pd.DataFrame:
    """Build the transaction-level feature table from a raw multi-event-type stream.

    Parameters
    ----------
    events:
        Every event (transactions, logins, authentication failures, ...) for
        however many users, with at least the columns produced by
        ``src/data/historical_generator.py``. Order does not matter — this
        function sorts internally, per user, before computing anything.

    Returns
    -------
    One row per transaction event, indexed like ``events`` was, with
    ``ID_COLUMNS``, every column in ``FEATURE_COLUMNS``, and — if present in
    the input — ``LABEL_COLUMN`` and ``episode_type`` carried through
    untouched for training/evaluation use.
    """

    events = events.copy()
    events["event_time"] = pd.to_datetime(events["event_time"])

    feature_frames = []
    for _user_id, group in events.groupby("user_id", sort=False):
        g = group.sort_values("event_time").reset_index(drop=True)
        feats = _user_features(g)
        feats = feats.join(g[["event_id", "user_id", "event_time"]])
        if "is_fraud_episode" in g.columns:
            feats = feats.join(g[["is_fraud_episode", "episode_type"]])
        feature_frames.append(feats)

    result = pd.concat(feature_frames, ignore_index=True)
    result = result.sort_values("event_time").reset_index(drop=True)

    ordered = ID_COLUMNS + FEATURE_COLUMNS
    trailing = [c for c in result.columns if c not in ordered]
    return result[ordered + trailing]
