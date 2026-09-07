"""Injects short, realistic fraud episodes into a user's own event timeline.

This is the *only* place ground truth comes from. An episode is a burst of
events layered on top of a user's normal history so that, relative to that
user's own past, it looks anomalous — new device, a burst of failed logins,
a velocity spike, unusual amounts. The label (``is_fraud_episode``) is a
property of the simulation (we know what we injected), never a function of
the engineered features themselves, so training on it is not leakage: it is
exactly analogous to a real system training on confirmed-fraud labels from
chargebacks while scoring on independently engineered behavioural features.

Two episode archetypes are implemented, mirroring two common real-world
patterns:

* ``account_takeover`` — a handful of failed logins, then a success from a
  brand-new device, then a few unusually large purchases in quick succession.
* ``card_testing`` — many very small, rapid-fire purchases (attackers probing
  whether a stolen card number is still live) with no login activity at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

import numpy as np

from src.data.user_population import LOCATIONS, SimUser
from src.producers.event_schema import TransactionEvent


def _episode_location(user: SimUser, rng: np.random.Generator, away_probability: float) -> str:
    """Usually the user's own city; sometimes somewhere they've never transacted from.

    A real account-takeover often shows up from a different region than the
    victim's usual one — without this, ``distinct_locations_24h`` would be a
    constant 1 for every user and carry no information at all.
    """
    if rng.random() < away_probability:
        others = [loc for loc in LOCATIONS if loc != user.home_location]
        return str(rng.choice(others))
    return user.home_location

EpisodeType = Literal["account_takeover", "card_testing"]


@dataclass(frozen=True)
class InjectedEvent:
    event: TransactionEvent
    is_fraud_episode: bool
    episode_type: str | None


def _uuid(rng: np.random.Generator) -> str:
    return f"evt_{rng.integers(0, 2**63 - 1):020d}"


def generate_account_takeover(
    user: SimUser,
    start_time: datetime,
    rng: np.random.Generator,
) -> list[InjectedEvent]:
    """A stolen-credential episode: failed logins, a new device, then a spend spree."""

    events: list[InjectedEvent] = []
    t = start_time
    # One location for the whole episode — a single attacker session, not a
    # different city per event.
    episode_location = _episode_location(user, rng, away_probability=0.55)

    # Not every takeover is noisy about it — a phished password can work on
    # the first try. Occasional zero-failure episodes stop "any failed
    # login at all" from being a free giveaway.
    n_failed = int(rng.integers(0, 6))
    for _ in range(n_failed):
        t += timedelta(seconds=float(rng.uniform(5, 40)))
        events.append(
            InjectedEvent(
                TransactionEvent(
                    event_id=_uuid(rng),
                    user_id=user.user_id,
                    event_time=t,
                    event_type="authentication_failure",
                    device_id=f"unknown_device_{rng.integers(1000, 9999)}",
                    location=episode_location,
                    session_id=f"session_{user.user_id}_{int(t.timestamp())}",
                    authentication_status="failure",
                ),
                is_fraud_episode=True,
                episode_type="account_takeover",
            )
        )

    # Usually a brand-new device — but a quarter of the time the attacker
    # reuses a device this user has genuinely used before (a shared family
    # laptop, a synced session), which is exactly what makes "new device"
    # alone an imperfect signal rather than a guaranteed tell.
    t += timedelta(seconds=float(rng.uniform(10, 60)))
    if rng.random() < 0.25:
        new_device = user.random_home_device(rng)
    else:
        new_device = f"fraud_device_{rng.integers(1000, 9999)}_{user.user_id}"
    events.append(
        InjectedEvent(
            TransactionEvent(
                event_id=_uuid(rng),
                user_id=user.user_id,
                event_time=t,
                event_type="login",
                device_id=new_device,
                location=episode_location,
                session_id=f"session_{user.user_id}_{int(t.timestamp())}",
                authentication_status="success",
            ),
            is_fraud_episode=True,
            episode_type="account_takeover",
        )
    )

    n_purchases = int(rng.integers(1, 5))
    for _ in range(n_purchases):
        t += timedelta(seconds=float(rng.uniform(15, 90)))
        # A wide, sometimes-modest multiplier: the top end is an obvious
        # spend spree, the bottom end overlaps with a merely large — but
        # perfectly legitimate — one-off purchase.
        multiplier = float(rng.uniform(1.3, 6.0))
        amount = float(np.clip(rng.normal(user.average_amount * multiplier, user.average_amount * 0.3), 200, None))
        events.append(
            InjectedEvent(
                TransactionEvent(
                    event_id=_uuid(rng),
                    user_id=user.user_id,
                    event_time=t,
                    event_type="transaction",
                    amount=round(amount, 2),
                    device_id=new_device,
                    location=episode_location,
                    session_id=f"session_{user.user_id}_{int(t.timestamp())}",
                    merchant_id=f"merchant_{rng.integers(100, 999)}",
                    transaction_type="online_purchase",
                    authentication_status="success",
                ),
                is_fraud_episode=True,
                episode_type="account_takeover",
            )
        )

    return events


def generate_card_testing(
    user: SimUser,
    start_time: datetime,
    rng: np.random.Generator,
) -> list[InjectedEvent]:
    """A card-testing episode: a rapid burst of small purchases, no login noise."""

    events: list[InjectedEvent] = []
    t = start_time
    device = user.random_home_device(rng) if rng.random() < 0.4 else f"fraud_device_{rng.integers(1000, 9999)}"
    episode_location = _episode_location(user, rng, away_probability=0.25)

    n_tests = int(rng.integers(3, 12))
    for _ in range(n_tests):
        t += timedelta(seconds=float(rng.uniform(2, 25)))
        amount = float(np.clip(rng.normal(150, 90), 10, 700))
        events.append(
            InjectedEvent(
                TransactionEvent(
                    event_id=_uuid(rng),
                    user_id=user.user_id,
                    event_time=t,
                    event_type="transaction",
                    amount=round(amount, 2),
                    device_id=device,
                    location=episode_location,
                    session_id=f"session_{user.user_id}_{int(t.timestamp())}",
                    merchant_id=f"merchant_{rng.integers(100, 999)}",
                    transaction_type="online_purchase",
                    authentication_status="success",
                ),
                is_fraud_episode=True,
                episode_type="card_testing",
            )
        )

    return events


def generate_episode(
    episode_type: EpisodeType,
    user: SimUser,
    start_time: datetime,
    rng: np.random.Generator,
) -> list[InjectedEvent]:
    if episode_type == "account_takeover":
        return generate_account_takeover(user, start_time, rng)
    if episode_type == "card_testing":
        return generate_card_testing(user, start_time, rng)
    raise ValueError(f"Unknown episode type: {episode_type}")
