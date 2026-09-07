"""The online counterpart to ``temporal_features.compute_features``.

Training reads a whole event history and slices windows out of it with
pandas. A live request can't do that — there is no dataframe, only "whatever
happened before now" — so this class keeps exactly the state needed to
answer the same questions (`FEATURE_COLUMNS`) incrementally, in Redis, and
computes them the same way: read the *prior* state, then record the new
event so the *next* call sees it.

Two calls, always in this order for a given transaction:

1. ``get_features(...)``  — read prior state, compute this transaction's
   feature vector (never touches state; safe to call more than once).
2. ``record_event(...)``  — write this event into that state, once, after
   scoring, so it becomes "prior history" for the user's next event.

Redis layout (all keys namespaced ``rt:*`` — "real-time", to keep this
clearly separate from Feast's own online store, which lives in Redis DB 1):

    rt:devices:{user_id}      SET     every device ever seen for this user
    rt:txn:{user_id}          ZSET    score=epoch, member="{seq}:{amount}"
    rt:auth_fail:{user_id}    ZSET    score=epoch, member="{seq}"
    rt:dev_events:{user_id}   ZSET    score=epoch, member="{seq}:{device_id}"
    rt:loc:{user_id}          ZSET    score=epoch, member="{seq}:{location}"
    rt:last_event_time:{u}    STRING  epoch of the user's last event
    rt:last_txn_time:{u}      STRING  epoch of the user's last transaction
    rt:event_count:{u}        STRING  running count of this user's events

Every key carries a TTL a little past the longest feature window (30 days),
so an inactive user's state ages out on its own rather than growing Redis
forever.
"""

from __future__ import annotations

import uuid
from datetime import datetime

import numpy as np

from src.features.temporal_features import LARGE_GAP_SECONDS, WINDOWS, cyclical

WINDOW_SECONDS = {name: int(delta / np.timedelta64(1, "s")) for name, delta in WINDOWS.items()}
HISTORY_TTL_SECONDS = WINDOW_SECONDS["30d"] + 5 * 86400


class FeatureStoreClient:
    def __init__(self, redis_client, namespace: str = "rt"):
        self.r = redis_client
        self.ns = namespace

    def _k(self, kind: str, user_id: str) -> str:
        return f"{self.ns}:{kind}:{user_id}"

    # ---- read: build this transaction's feature vector from prior state ----

    def get_features(
        self,
        user_id: str,
        event_time: datetime,
        amount: float,
        device_id: str,
        location: str,
    ) -> dict:
        now = event_time.timestamp()
        upper = f"({now}"  # Redis range syntax: exclusive upper bound

        is_new_device = 0 if self.r.sismember(self._k("devices", user_id), device_id) else 1

        count_5m, _, _ = self._txn_window_stats(user_id, now - WINDOW_SECONDS["5m"], upper)
        count_1h, _, _ = self._txn_window_stats(user_id, now - WINDOW_SECONDS["1h"], upper)
        count_24h, mean_24h, std_24h = self._txn_window_stats(user_id, now - WINDOW_SECONDS["24h"], upper)
        count_30d, mean_30d, std_30d = self._txn_window_stats(user_id, now - WINDOW_SECONDS["30d"], upper)

        if count_30d >= 3:
            std_floor = max(std_30d, 0.05 * mean_30d, 1.0)
            zscore = (amount - mean_30d) / std_floor
        else:
            zscore = 0.0

        auth_1h = self.r.zcount(self._k("auth_fail", user_id), now - WINDOW_SECONDS["1h"], upper)
        auth_24h = self.r.zcount(self._k("auth_fail", user_id), now - WINDOW_SECONDS["24h"], upper)

        window_devices = self._members_in_window(self._k("dev_events", user_id), now - WINDOW_SECONDS["24h"], upper)
        window_locations = self._members_in_window(self._k("loc", user_id), now - WINDOW_SECONDS["24h"], upper)
        unique_devices_24h = len({m.rsplit(":", 1)[-1] for m in window_devices})
        distinct_locations_24h = len({m.rsplit(":", 1)[-1] for m in window_locations}) if window_locations else 1

        last_event_time = self.r.get(self._k("last_event_time", user_id))
        seconds_since_last_event = now - float(last_event_time) if last_event_time else LARGE_GAP_SECONDS
        last_txn_time = self.r.get(self._k("last_txn_time", user_id))
        seconds_since_last_txn = now - float(last_txn_time) if last_txn_time else LARGE_GAP_SECONDS

        prior_event_count = int(self.r.get(self._k("event_count", user_id)) or 0)

        hour = event_time.hour + event_time.minute / 60.0
        dow = event_time.weekday()
        hour_sin, hour_cos = cyclical(hour, 24.0)
        dow_sin, dow_cos = cyclical(dow, 7.0)

        return {
            "amount": amount,
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
            "unique_devices_24h": unique_devices_24h,
            "is_new_device": is_new_device,
            "auth_failure_count_1h": auth_1h,
            "auth_failure_count_24h": auth_24h,
            "distinct_locations_24h": distinct_locations_24h,
            "user_prior_event_count": prior_event_count,
            "hour_sin": float(hour_sin),
            "hour_cos": float(hour_cos),
            "dow_sin": float(dow_sin),
            "dow_cos": float(dow_cos),
        }

    def _txn_window_stats(self, user_id: str, lower: float, upper: str) -> tuple[int, float, float]:
        raw = self._members_in_window(self._k("txn", user_id), lower, upper)
        if not raw:
            return 0, 0.0, 0.0
        amounts = np.array([float(m.rsplit(":", 1)[-1]) for m in raw])
        return len(amounts), float(amounts.mean()), float(amounts.std())

    def _members_in_window(self, key: str, lower: float, upper: str) -> list[str]:
        return list(self.r.zrangebyscore(key, lower, upper))

    # ---- write: record this event as history for the *next* call ----

    def record_event(
        self,
        user_id: str,
        event_time: datetime,
        event_type: str,
        amount: float | None,
        device_id: str,
        location: str,
    ) -> None:
        now = event_time.timestamp()
        seq = uuid.uuid4().hex[:12]  # disambiguates same-timestamp events

        pipe = self.r.pipeline()

        devices_key = self._k("devices", user_id)
        pipe.sadd(devices_key, device_id)
        pipe.expire(devices_key, HISTORY_TTL_SECONDS)

        if event_type == "transaction" and amount is not None:
            txn_key = self._k("txn", user_id)
            pipe.zadd(txn_key, {f"{seq}:{amount}": now})
            pipe.zremrangebyscore(txn_key, "-inf", now - WINDOW_SECONDS["30d"])
            pipe.expire(txn_key, HISTORY_TTL_SECONDS)
            pipe.set(self._k("last_txn_time", user_id), now, ex=HISTORY_TTL_SECONDS)

        if event_type == "authentication_failure":
            auth_key = self._k("auth_fail", user_id)
            pipe.zadd(auth_key, {seq: now})
            pipe.zremrangebyscore(auth_key, "-inf", now - WINDOW_SECONDS["24h"])
            pipe.expire(auth_key, HISTORY_TTL_SECONDS)

        dev_win_key = self._k("dev_events", user_id)
        pipe.zadd(dev_win_key, {f"{seq}:{device_id}": now})
        pipe.zremrangebyscore(dev_win_key, "-inf", now - WINDOW_SECONDS["24h"])
        pipe.expire(dev_win_key, HISTORY_TTL_SECONDS)

        loc_key = self._k("loc", user_id)
        pipe.zadd(loc_key, {f"{seq}:{location}": now})
        pipe.zremrangebyscore(loc_key, "-inf", now - WINDOW_SECONDS["24h"])
        pipe.expire(loc_key, HISTORY_TTL_SECONDS)

        pipe.set(self._k("last_event_time", user_id), now, ex=HISTORY_TTL_SECONDS)
        pipe.incr(self._k("event_count", user_id))
        pipe.expire(self._k("event_count", user_id), HISTORY_TTL_SECONDS)

        pipe.execute()
