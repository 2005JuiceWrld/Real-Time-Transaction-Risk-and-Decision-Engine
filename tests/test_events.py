from datetime import datetime

import pytest

from src.data.user_population import SimUser, build_population
from src.producers.event_generator import EventGenerator
from src.producers.event_schema import TransactionEvent
from src.producers.scenarios import ALL_PROFILES, NORMAL_USER


def test_transaction_event_creation():
    event = TransactionEvent(
        event_id="evt_001",
        user_id="user_001",
        event_time=datetime(2026, 9, 7, 12, 0, 0),
        event_type="transaction",
        amount=1500.0,
        device_id="device_001",
        location="Mumbai",
        session_id="session_001",
        merchant_id="merchant_001",
        transaction_type="online_purchase",
        authentication_status="success",
    )

    assert event.event_id == "evt_001"
    assert event.user_id == "user_001"
    assert event.amount == 1500.0


def test_negative_amount_is_rejected():
    with pytest.raises(ValueError):
        TransactionEvent(
            event_id="evt_002",
            user_id="user_001",
            event_time=datetime(2026, 9, 7, 12, 0, 0),
            event_type="transaction",
            amount=-100,
            device_id="device_001",
            location="Mumbai",
            session_id="session_001",
        )


def _make_user(**overrides) -> SimUser:
    defaults = dict(
        user_id="user_001",
        profile=NORMAL_USER,
        average_amount=1200.0,
        amount_variability=0.3,
        transactions_per_day=0.8,
        active_hour_center=14.0,
        active_hour_spread=4.0,
        authentication_failure_rate=0.01,
        home_location="Mumbai",
        devices=("user_001_dev_0",),
    )
    defaults.update(overrides)
    return SimUser(**defaults)


def test_event_generator_transaction():
    generator = EventGenerator(seed=1)
    user = _make_user()

    event = generator.generate_transaction(user, datetime(2026, 9, 7, 12, 0, 0))

    assert event.user_id == "user_001"
    assert event.event_type == "transaction"
    assert event.amount is not None and event.amount > 0
    assert event.device_id == "user_001_dev_0"


def test_event_generator_login_can_fail():
    generator = EventGenerator(seed=1)
    # authentication_failure_rate=1.0 forces every login to fail.
    user = _make_user(authentication_failure_rate=1.0)

    event = generator.generate_login(user, datetime(2026, 9, 7, 12, 0, 0))

    assert event.event_type == "authentication_failure"
    assert event.authentication_status == "failure"


def test_event_generator_login_forced_failure():
    generator = EventGenerator(seed=1)
    user = _make_user(authentication_failure_rate=0.0)

    event = generator.generate_login(user, datetime(2026, 9, 7, 12, 0, 0), force_failure=True)

    assert event.event_type == "authentication_failure"


def test_profiles_have_population_shares_summing_near_one():
    assert len(ALL_PROFILES) >= 3
    total_share = sum(p.population_share for p in ALL_PROFILES)
    assert 0.9 <= total_share <= 1.1


def test_build_population_is_deterministic():
    pop_a = build_population(n_users=50, seed=7)
    pop_b = build_population(n_users=50, seed=7)

    assert [u.user_id for u in pop_a] == [u.user_id for u in pop_b]
    assert [u.average_amount for u in pop_a] == [u.average_amount for u in pop_b]
    assert [u.devices for u in pop_a] == [u.devices for u in pop_b]


def test_build_population_gives_each_user_at_least_one_device():
    population = build_population(n_users=30, seed=3)
    for user in population:
        assert len(user.devices) >= 1
        assert all(d.startswith(user.user_id) for d in user.devices)
