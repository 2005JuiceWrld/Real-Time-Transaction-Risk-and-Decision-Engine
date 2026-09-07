"""Builds a reproducible, chronologically realistic historical event dataset.

Unlike a per-event random label, this generator gives every synthetic user a
persistent identity (``src/data/user_population.py``) and a full chronological
event timeline of their own — transactions, logins, occasional failed
authentications — produced by a non-homogeneous Poisson arrival process
modulated by that user's own active hours. A small fraction of users then get
a short fraud episode (``src/producers/fraud_injection.py``) spliced into
their timeline.

The output keeps ``is_fraud_episode`` / ``episode_type`` as label columns only.
Nothing in ``src/features/temporal_features.py`` is allowed to read them —
features are computed purely from the event stream itself.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from src.data.user_population import SimUser, build_population
from src.producers.event_generator import EventGenerator
from src.producers.event_schema import TransactionEvent
from src.producers.fraud_injection import InjectedEvent, generate_episode


def _hour_intensity(hours: np.ndarray, center: float, spread: float) -> np.ndarray:
    """A circular (24h-wrapped) Gaussian bump used to shape a user's activity by time of day."""
    diff = np.minimum(np.abs(hours - center), 24 - np.abs(hours - center))
    return np.exp(-0.5 * (diff / spread) ** 2)


class HistoricalDataGenerator:
    """Generate a reproducible historical multi-user event dataset."""

    def __init__(
        self,
        seed: int = 42,
        n_days: int = 60,
        fraud_rate: float = 0.045,
        start_date: datetime = datetime(2026, 1, 1),
    ):
        self.seed = seed
        self.n_days = n_days
        self.fraud_rate = fraud_rate
        self.start_date = start_date
        self.rng = np.random.default_rng(seed)
        self.event_gen = EventGenerator(seed=seed + 1)

    def _arrival_times(self, user: SimUser) -> list[datetime]:
        """Thinned non-homogeneous Poisson process over the simulation window."""

        window_seconds = self.n_days * 24 * 3600
        max_rate_per_second = (user.transactions_per_day / 86400.0) * 3.0

        expected_candidates = max_rate_per_second * window_seconds
        n_candidates = self.rng.poisson(max(expected_candidates, 1))

        offsets = np.sort(self.rng.uniform(0, window_seconds, size=n_candidates))
        candidate_times = [self.start_date + timedelta(seconds=float(s)) for s in offsets]

        hours = np.array([t.hour + t.minute / 60.0 for t in candidate_times])
        intensity = _hour_intensity(hours, user.active_hour_center, user.active_hour_spread)
        accept = self.rng.random(len(candidate_times)) < intensity

        return [t for t, keep in zip(candidate_times, accept) if keep]

    def _normal_timeline(self, user: SimUser) -> list[InjectedEvent]:
        events: list[InjectedEvent] = []
        for t in self._arrival_times(user):
            if self.rng.random() < 0.12:
                # Most login attempts just work. A small fraction of genuine
                # users fat-finger their password once or twice before
                # succeeding — a benign failed-login burst that overlaps,
                # on purpose, with what a fraud episode also looks like.
                if self.rng.random() < 0.04:
                    n_typos = int(self.rng.integers(1, 3))
                    for _ in range(n_typos):
                        typo_time = t - timedelta(seconds=float(self.rng.uniform(5, 30)) * (n_typos))
                        events.append(
                            InjectedEvent(
                                event=self.event_gen.generate_login(user, typo_time, force_failure=True),
                                is_fraud_episode=False,
                                episode_type=None,
                            )
                        )
                event = self.event_gen.generate_login(user, t)
                events.append(InjectedEvent(event=event, is_fraud_episode=False, episode_type=None))
            else:
                event = self.event_gen.generate_transaction(user, t)
                events.append(InjectedEvent(event=event, is_fraud_episode=False, episode_type=None))

                # Real shopping isn't always one purchase at a time — paying
                # separate sellers, retrying a declined card, splitting a
                # cart. A genuine rapid-fire burst, benign, is what keeps
                # "several transactions within minutes" from being a
                # perfect tell on its own — velocity alone dominated every
                # earlier version of this model precisely because normal
                # users never did this.
                if self.rng.random() < 0.05:
                    burst_time = t
                    for _ in range(int(self.rng.integers(1, 4))):
                        burst_time += timedelta(seconds=float(self.rng.uniform(10, 180)))
                        events.append(
                            InjectedEvent(
                                event=self.event_gen.generate_transaction(user, burst_time),
                                is_fraud_episode=False,
                                episode_type=None,
                            )
                        )
        return events

    def _maybe_inject_fraud(
        self,
        user: SimUser,
        timeline: list[InjectedEvent],
    ) -> list[InjectedEvent]:
        if len(timeline) < 6 or self.rng.random() >= self.fraud_rate:
            return timeline

        # Inject somewhere in the middle 70% of the user's own history, so
        # there is always genuine prior behaviour to compute rolling features
        # from and always some "after" period too.
        lo = int(len(timeline) * 0.20)
        hi = int(len(timeline) * 0.90)
        if hi <= lo:
            return timeline
        anchor_idx = int(self.rng.integers(lo, hi))
        anchor_time = timeline[anchor_idx].event.event_time + timedelta(seconds=float(self.rng.uniform(30, 300)))

        episode_type = "account_takeover" if self.rng.random() < 0.6 else "card_testing"
        episode_events = generate_episode(episode_type, user, anchor_time, self.rng)

        return timeline + episode_events

    def _user_events(self, user: SimUser) -> list[InjectedEvent]:
        timeline = self._normal_timeline(user)
        timeline = self._maybe_inject_fraud(user, timeline)
        timeline.sort(key=lambda item: item.event.event_time)
        return timeline

    def generate(self, n_users: int = 500) -> pd.DataFrame:
        population = build_population(n_users=n_users, seed=self.seed)

        rows: list[dict] = []
        for user in population:
            for item in self._user_events(user):
                rows.append(_event_to_row(item))

        df = pd.DataFrame(rows)
        df = df.sort_values("event_time").reset_index(drop=True)
        return df


def _event_to_row(item: InjectedEvent) -> dict:
    e: TransactionEvent = item.event
    return {
        "event_id": e.event_id,
        "user_id": e.user_id,
        "event_time": e.event_time,
        "event_type": e.event_type,
        "amount": e.amount,
        "currency": e.currency,
        "device_id": e.device_id,
        "location": e.location,
        "session_id": e.session_id,
        "merchant_id": e.merchant_id,
        "transaction_type": e.transaction_type,
        "authentication_status": e.authentication_status,
        # Label columns — simulation ground truth only. Never used as model
        # inputs; see src/features/temporal_features.py.
        "is_fraud_episode": item.is_fraud_episode,
        "episode_type": item.episode_type,
    }
