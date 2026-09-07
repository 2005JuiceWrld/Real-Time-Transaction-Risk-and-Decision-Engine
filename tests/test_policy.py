import json

import pytest

from src.decisions.decision_log import DecisionLogger
from src.decisions.policy import ALLOW, BLOCK, REVIEW, DecisionEngine, PolicyConfig, ReasonRule


@pytest.fixture
def engine():
    config = PolicyConfig(
        allow_below=0.30,
        block_at_or_above=0.75,
        reason_rules=(
            ReasonRule(code="NEW_DEVICE", feature="is_new_device", op=">=", value=1),
            ReasonRule(code="HIGH_VELOCITY", feature="txn_count_5m", op=">=", value=3),
        ),
        fallback_reason_code="MODEL_HIGH_RISK",
    )
    return DecisionEngine(config, model_version="test-v1")


def test_low_risk_is_allowed(engine):
    record = engine.decide("evt_1", "user_1", risk_score=0.05, features={"is_new_device": 0, "txn_count_5m": 0})
    assert record.decision == ALLOW
    assert record.reason_codes == []


def test_mid_risk_is_reviewed(engine):
    record = engine.decide("evt_2", "user_1", risk_score=0.5, features={"is_new_device": 1, "txn_count_5m": 0})
    assert record.decision == REVIEW
    assert "NEW_DEVICE" in record.reason_codes


def test_high_risk_is_blocked(engine):
    record = engine.decide(
        "evt_3", "user_1", risk_score=0.9, features={"is_new_device": 1, "txn_count_5m": 5}
    )
    assert record.decision == BLOCK
    assert set(record.reason_codes) == {"NEW_DEVICE", "HIGH_VELOCITY"}


def test_boundaries_are_inclusive_to_the_higher_tier(engine):
    # exactly at allow_below -> REVIEW, not ALLOW
    assert engine.classify(0.30) == REVIEW
    # exactly at block_at_or_above -> BLOCK
    assert engine.classify(0.75) == BLOCK
    assert engine.classify(0.2999) == ALLOW


def test_review_or_block_with_no_matching_rule_gets_fallback_code(engine):
    record = engine.decide(
        "evt_4", "user_1", risk_score=0.8, features={"is_new_device": 0, "txn_count_5m": 0}
    )
    assert record.decision == BLOCK
    assert record.reason_codes == ["MODEL_HIGH_RISK"]


def test_allow_can_still_carry_an_informational_reason_code(engine):
    # Reason codes describe what was *observed* (useful for audit even on an
    # ALLOW — "allowed despite a new device because overall risk was low");
    # only the fallback code is gated on the decision being REVIEW/BLOCK.
    record = engine.decide(
        "evt_5", "user_1", risk_score=0.05, features={"is_new_device": 1, "txn_count_5m": 0}
    )
    assert record.decision == ALLOW
    assert record.reason_codes == ["NEW_DEVICE"]


def test_allow_with_no_matching_rule_gets_no_fallback_code(engine):
    record = engine.decide("evt_5b", "user_1", risk_score=0.05, features={"is_new_device": 0, "txn_count_5m": 0})
    assert record.decision == ALLOW
    assert record.reason_codes == []


def test_missing_feature_does_not_crash_rule_evaluation(engine):
    record = engine.decide("evt_6", "user_1", risk_score=0.5, features={})
    assert record.decision == REVIEW
    assert record.reason_codes == ["MODEL_HIGH_RISK"]


def test_policy_loads_from_the_real_config_file():
    config = PolicyConfig.from_yaml()
    assert 0.0 <= config.allow_below < config.block_at_or_above <= 1.0
    assert len(config.reason_rules) > 0


def test_decision_logger_writes_one_json_line_per_record(tmp_path):
    log_path = tmp_path / "decisions.jsonl"
    logger = DecisionLogger(path=log_path)

    engine = DecisionEngine(
        PolicyConfig(allow_below=0.3, block_at_or_above=0.75, reason_rules=()),
    )
    logger.log(engine.decide("evt_1", "user_1", 0.1, {}))
    logger.log(engine.decide("evt_2", "user_1", 0.9, {}))

    lines = log_path.read_text().strip().split("\n")
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["event_id"] == "evt_1"
    assert first["decision"] == "ALLOW"
