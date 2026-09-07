from datetime import datetime

import numpy as np
import pytest

from src.data.user_population import build_population
from src.producers.fraud_injection import generate_account_takeover, generate_card_testing, generate_episode


@pytest.fixture
def user():
    return build_population(n_users=1, seed=42)[0]


def test_account_takeover_events_are_all_labelled_fraud(user):
    rng = np.random.default_rng(1)
    events = generate_account_takeover(user, datetime(2026, 1, 15, 10, 0, 0), rng)

    assert len(events) > 0
    assert all(e.is_fraud_episode for e in events)
    assert all(e.episode_type == "account_takeover" for e in events)


def test_account_takeover_events_are_chronological(user):
    rng = np.random.default_rng(2)
    events = generate_account_takeover(user, datetime(2026, 1, 15, 10, 0, 0), rng)

    times = [e.event.event_time for e in events]
    assert times == sorted(times)


def test_account_takeover_includes_a_purchase_from_a_device_used_in_the_login(user):
    rng = np.random.default_rng(3)
    events = generate_account_takeover(user, datetime(2026, 1, 15, 10, 0, 0), rng)

    logins = [e for e in events if e.event.event_type == "login"]
    purchases = [e for e in events if e.event.event_type == "transaction"]

    assert len(logins) == 1
    assert len(purchases) >= 1
    assert all(p.event.device_id == logins[0].event.device_id for p in purchases)


def test_card_testing_events_are_small_and_rapid(user):
    rng = np.random.default_rng(4)
    events = generate_card_testing(user, datetime(2026, 1, 15, 10, 0, 0), rng)

    assert all(e.event.event_type == "transaction" for e in events)
    assert all(e.event.amount is not None and e.event.amount <= 700 for e in events)

    times = [e.event.event_time for e in events]
    total_span = (times[-1] - times[0]).total_seconds()
    assert total_span < 400  # a burst, not spread over the day


def test_generate_episode_dispatches_by_type(user):
    rng = np.random.default_rng(5)
    takeover = generate_episode("account_takeover", user, datetime(2026, 1, 15), rng)
    card_testing = generate_episode("card_testing", user, datetime(2026, 1, 15), rng)

    assert all(e.episode_type == "account_takeover" for e in takeover)
    assert all(e.episode_type == "card_testing" for e in card_testing)


def test_generate_episode_rejects_unknown_type(user):
    rng = np.random.default_rng(6)
    with pytest.raises(ValueError):
        generate_episode("not_a_real_episode", user, datetime(2026, 1, 15), rng)  # type: ignore[arg-type]


def test_episodes_are_reproducible_given_the_same_rng_state(user):
    events_a = generate_account_takeover(user, datetime(2026, 1, 15), np.random.default_rng(99))
    events_b = generate_account_takeover(user, datetime(2026, 1, 15), np.random.default_rng(99))

    amounts_a = [e.event.amount for e in events_a]
    amounts_b = [e.event.amount for e in events_b]
    assert amounts_a == amounts_b
