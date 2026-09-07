"""Persistent user archetypes.

Each archetype is a *baseline behavioural template* assigned to a user for
their entire lifetime in the simulation — it is never exposed to the model.
It only shapes how a user's own event stream looks (how often they transact,
how much they spend, how many devices they own, how sloppy they are with
logins). Genuine fraud/anomaly signal comes later, from short injected
"fraud episodes" layered on top of a user's own history
(see ``src/producers/fraud_injection.py``), not from the archetype itself.

Splitting "what a user is normally like" (this file) from "what happens
during an attack" (fraud_injection.py) is what makes it possible to compute
honest, causal temporal features: a rolling window over a user's own past
events reflects their own baseline, and an injected episode reads as an
anomaly relative to *that specific user*, the way real fraud detection
actually works — not because a hidden label leaked into the feature table.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class UserProfile:
    name: str
    average_transaction_amount: float
    amount_variability: float
    """Coefficient of variation (std / mean) for this user's transaction amounts."""
    transactions_per_day: float
    """Mean number of transactions/day (Poisson rate) for a chronological arrival process."""
    device_count: int
    """How many distinct devices this user normally rotates between."""
    authentication_failure_rate: float
    """Probability an authentication event for this user fails, at baseline."""
    active_hour_center: float
    """Hour of day (0-23, may be fractional) this user is usually active around."""
    active_hour_spread: float
    """Std-dev, in hours, of this user's activity around their active-hour centre."""
    population_share: float
    """Roughly what fraction of the synthetic population gets this archetype."""


NORMAL_USER = UserProfile(
    name="normal",
    average_transaction_amount=1200,
    amount_variability=0.30,
    transactions_per_day=0.8,
    device_count=1,
    authentication_failure_rate=0.01,
    active_hour_center=14.0,
    active_hour_spread=4.5,
    population_share=0.55,
)


HIGH_VALUE_USER = UserProfile(
    name="high_value",
    average_transaction_amount=15000,
    amount_variability=0.35,
    transactions_per_day=0.4,
    device_count=2,
    authentication_failure_rate=0.01,
    active_hour_center=13.0,
    active_hour_spread=3.5,
    population_share=0.10,
)


HIGH_VELOCITY_USER = UserProfile(
    name="high_velocity",
    average_transaction_amount=900,
    amount_variability=0.40,
    transactions_per_day=4.5,
    device_count=2,
    authentication_failure_rate=0.02,
    active_hour_center=15.0,
    active_hour_spread=5.5,
    population_share=0.12,
)


DEVICE_SWITCHING_USER = UserProfile(
    name="device_switching",
    average_transaction_amount=1800,
    amount_variability=0.45,
    transactions_per_day=1.2,
    device_count=3,
    authentication_failure_rate=0.03,
    active_hour_center=16.0,
    active_hour_spread=6.0,
    population_share=0.13,
)


NIGHT_OWL_USER = UserProfile(
    name="night_owl",
    average_transaction_amount=1400,
    amount_variability=0.35,
    transactions_per_day=0.7,
    device_count=1,
    authentication_failure_rate=0.015,
    active_hour_center=23.5,
    active_hour_spread=3.0,
    population_share=0.10,
)


LOW_ACTIVITY_USER = UserProfile(
    name="low_activity",
    average_transaction_amount=800,
    amount_variability=0.25,
    transactions_per_day=0.15,
    device_count=1,
    authentication_failure_rate=0.01,
    active_hour_center=12.0,
    active_hour_spread=4.0,
    population_share=0.00,
)

ALL_PROFILES = (
    NORMAL_USER,
    HIGH_VALUE_USER,
    HIGH_VELOCITY_USER,
    DEVICE_SWITCHING_USER,
    NIGHT_OWL_USER,
)
