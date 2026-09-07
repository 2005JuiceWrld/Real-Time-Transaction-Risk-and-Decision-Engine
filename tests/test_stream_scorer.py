from datetime import datetime, timedelta, timezone

import fakeredis

from src.decisions.decision_log import DecisionLogger
from src.decisions.policy import DecisionEngine, PolicyConfig, ReasonRule
from src.features.feature_store_client import FeatureStoreClient
from src.models.infer import RiskModel
from src.serving.stream_scorer import _handle


def _event(event_id, user_id="u1", event_type="transaction", amount=500.0, device_id="dev_1",
           location="Mumbai", event_time=None):
    return {
        "event_id": event_id,
        "user_id": user_id,
        "event_time": (event_time or datetime.now(timezone.utc)).isoformat(),
        "event_type": event_type,
        "amount": amount,
        "device_id": device_id,
        "location": location,
        "session_id": f"session_{user_id}",
    }


def _engine():
    config = PolicyConfig(
        allow_below=0.3,
        block_at_or_above=0.75,
        reason_rules=(ReasonRule(code="NEW_DEVICE", feature="is_new_device", op=">=", value=1),),
    )
    return DecisionEngine(config)


def test_handle_scores_a_transaction_and_records_state(tmp_path):
    model = RiskModel.load()
    engine = _engine()
    decision_logger = DecisionLogger(path=tmp_path / "decisions.jsonl")
    store = FeatureStoreClient(fakeredis.FakeRedis(decode_responses=True))

    _handle(_event("e1"), model, engine, decision_logger, store)

    lines = (tmp_path / "decisions.jsonl").read_text().strip().split("\n")
    assert len(lines) == 1

    # The next transaction for the same user should see the first as history.
    features = store.get_features("u1", datetime.now(timezone.utc), 100.0, "dev_1", "Mumbai")
    assert features["user_prior_event_count"] == 1


def test_handle_ignores_malformed_messages_without_raising(tmp_path):
    model = RiskModel.load()
    engine = _engine()
    decision_logger = DecisionLogger(path=tmp_path / "decisions.jsonl")
    store = FeatureStoreClient(fakeredis.FakeRedis(decode_responses=True))

    _handle({"not": "a valid event"}, model, engine, decision_logger, store)

    assert not (tmp_path / "decisions.jsonl").exists()


def test_handle_updates_state_for_non_transaction_events_without_scoring(tmp_path):
    model = RiskModel.load()
    engine = _engine()
    decision_logger = DecisionLogger(path=tmp_path / "decisions.jsonl")
    store = FeatureStoreClient(fakeredis.FakeRedis(decode_responses=True))

    _handle(_event("e1", event_type="login", amount=None), model, engine, decision_logger, store)

    assert not (tmp_path / "decisions.jsonl").exists()  # logins are not decided on
    features = store.get_features("u1", datetime.now(timezone.utc), 100.0, "dev_1", "Mumbai")
    assert features["user_prior_event_count"] == 1  # but they still count as history


def test_handle_sequence_reflects_growing_history(tmp_path):
    model = RiskModel.load()
    engine = _engine()
    decision_logger = DecisionLogger(path=tmp_path / "decisions.jsonl")
    store = FeatureStoreClient(fakeredis.FakeRedis(decode_responses=True))

    t0 = datetime.now(timezone.utc)
    for i in range(3):
        _handle(_event(f"e{i}", event_time=t0 + timedelta(minutes=i)), model, engine, decision_logger, store)

    lines = (tmp_path / "decisions.jsonl").read_text().strip().split("\n")
    assert len(lines) == 3
