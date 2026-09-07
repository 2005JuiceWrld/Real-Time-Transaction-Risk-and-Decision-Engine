"""Builds a synthetic population of users with persistent, individual behaviour.

Two users with the same archetype (see ``src/producers/scenarios.py``) are not
identical: each gets its own jittered spending level, its own device IDs, its
own home location and its own active-hours centre. That individual variation
matters — without it, "amount" alone would trivially separate archetypes and
any model would learn the archetype rather than genuine behavioural signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.producers.scenarios import ALL_PROFILES, UserProfile

LOCATIONS = (
    "Mumbai",
    "Delhi",
    "Bengaluru",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Kolkata",
    "Ahmedabad",
)


@dataclass(frozen=True)
class SimUser:
    """One synthetic user's persistent, per-user baseline state."""

    user_id: str
    profile: UserProfile

    # Per-user jitter around the archetype's baseline, fixed for this user's
    # lifetime so a user's own history is internally consistent.
    average_amount: float
    amount_variability: float
    transactions_per_day: float
    active_hour_center: float
    active_hour_spread: float
    authentication_failure_rate: float

    home_location: str
    devices: tuple[str, ...]
    """This user's own device pool. The *first* device is their primary one."""

    def random_home_device(self, rng: np.random.Generator) -> str:
        """A device drawn from this user's own pool, weighted toward the primary one."""
        if len(self.devices) == 1:
            return self.devices[0]
        weights = np.array([0.7] + [0.3 / (len(self.devices) - 1)] * (len(self.devices) - 1))
        return str(rng.choice(self.devices, p=weights))


def _sample_profile(rng: np.random.Generator) -> UserProfile:
    shares = np.array([p.population_share for p in ALL_PROFILES], dtype=float)
    shares = shares / shares.sum()
    return ALL_PROFILES[int(rng.choice(len(ALL_PROFILES), p=shares))]


def build_population(n_users: int, seed: int = 42) -> list[SimUser]:
    """Deterministically build ``n_users`` synthetic users."""

    rng = np.random.default_rng(seed)
    users: list[SimUser] = []

    for index in range(n_users):
        profile = _sample_profile(rng)
        user_id = f"user_{index:05d}"

        # Log-normal jitter keeps amounts positive and right-skewed, like real
        # spending distributions, while staying centred on the archetype.
        average_amount = float(
            np.clip(rng.lognormal(mean=np.log(profile.average_transaction_amount), sigma=0.25), 100, None)
        )
        amount_variability = float(np.clip(rng.normal(profile.amount_variability, 0.05), 0.10, 0.80))
        transactions_per_day = float(
            np.clip(rng.lognormal(mean=np.log(max(profile.transactions_per_day, 0.02)), sigma=0.30), 0.02, None)
        )
        active_hour_center = float(np.mod(rng.normal(profile.active_hour_center, 1.5), 24.0))
        active_hour_spread = float(np.clip(rng.normal(profile.active_hour_spread, 1.0), 1.5, 8.0))
        auth_failure_rate = float(np.clip(rng.normal(profile.authentication_failure_rate, 0.005), 0.001, 0.5))

        home_location = str(rng.choice(LOCATIONS))
        devices = tuple(f"{user_id}_dev_{d}" for d in range(profile.device_count))

        users.append(
            SimUser(
                user_id=user_id,
                profile=profile,
                average_amount=average_amount,
                amount_variability=amount_variability,
                transactions_per_day=transactions_per_day,
                active_hour_center=active_hour_center,
                active_hour_spread=active_hour_spread,
                authentication_failure_rate=auth_failure_rate,
                home_location=home_location,
                devices=devices,
            )
        )

    return users
