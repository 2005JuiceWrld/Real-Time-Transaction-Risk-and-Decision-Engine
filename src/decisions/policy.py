"""The decision engine: turns a risk score + feature snapshot into ALLOW/REVIEW/BLOCK.

Thresholds and reason-code rules live in ``configs/decision_policy.yaml`` —
data, not code — so retuning the policy never requires touching this file or
redeploying the scoring path around it. Reason codes are evaluated
independently of the model, from simple, auditable feature-threshold rules;
that keeps "why was this reviewed?" answerable without inspecting the model
itself, and still gives a reason even for a borderline score a raw
probability alone can't explain.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import yaml

Decision = str  # one of "ALLOW", "REVIEW", "BLOCK"

ALLOW: Decision = "ALLOW"
REVIEW: Decision = "REVIEW"
BLOCK: Decision = "BLOCK"

_OPERATORS = {
    ">=": operator.ge,
    ">": operator.gt,
    "<=": operator.le,
    "<": operator.lt,
    "==": operator.eq,
    "!=": operator.ne,
}

DEFAULT_POLICY_PATH = Path("configs/decision_policy.yaml")


@dataclass(frozen=True)
class ReasonRule:
    code: str
    feature: str
    op: str
    value: float
    description: str = ""

    def matches(self, features: dict) -> bool:
        if self.feature not in features:
            return False
        compare = _OPERATORS[self.op]
        return bool(compare(features[self.feature], self.value))


@dataclass(frozen=True)
class PolicyConfig:
    allow_below: float
    block_at_or_above: float
    reason_rules: tuple[ReasonRule, ...]
    fallback_reason_code: str = "MODEL_HIGH_RISK"

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_POLICY_PATH) -> "PolicyConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)

        rules = tuple(
            ReasonRule(
                code=r["code"],
                feature=r["feature"],
                op=r["op"],
                value=r["value"],
                description=r.get("description", ""),
            )
            for r in raw.get("reason_codes", [])
        )

        return cls(
            allow_below=float(raw["thresholds"]["allow_below"]),
            block_at_or_above=float(raw["thresholds"]["block_at_or_above"]),
            reason_rules=rules,
            fallback_reason_code=raw.get("fallback_reason_code", "MODEL_HIGH_RISK"),
        )


@dataclass(frozen=True)
class DecisionRecord:
    event_id: str
    user_id: str
    risk_score: float
    decision: Decision
    reason_codes: list[str]
    timestamp: str
    model_version: str | None = None

    def to_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "user_id": self.user_id,
            "risk_score": self.risk_score,
            "decision": self.decision,
            "reason_codes": self.reason_codes,
            "timestamp": self.timestamp,
            "model_version": self.model_version,
        }


class DecisionEngine:
    def __init__(self, config: PolicyConfig, model_version: str | None = None):
        self.config = config
        self.model_version = model_version

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_POLICY_PATH, model_version: str | None = None) -> "DecisionEngine":
        return cls(PolicyConfig.from_yaml(path), model_version=model_version)

    def classify(self, risk_score: float) -> Decision:
        if risk_score >= self.config.block_at_or_above:
            return BLOCK
        if risk_score >= self.config.allow_below:
            return REVIEW
        return ALLOW

    def reason_codes_for(self, features: dict) -> list[str]:
        return [rule.code for rule in self.config.reason_rules if rule.matches(features)]

    def decide(
        self,
        event_id: str,
        user_id: str,
        risk_score: float,
        features: dict,
    ) -> DecisionRecord:
        decision = self.classify(risk_score)
        reasons = self.reason_codes_for(features)

        # A REVIEW/BLOCK with no matching rule means the model saw a
        # combination of signals no single threshold captures — say so,
        # rather than leaving the decision unexplained.
        if decision != ALLOW and not reasons:
            reasons = [self.config.fallback_reason_code]

        return DecisionRecord(
            event_id=event_id,
            user_id=user_id,
            risk_score=risk_score,
            decision=decision,
            reason_codes=reasons,
            timestamp=datetime.now(timezone.utc).isoformat(),
            model_version=self.model_version,
        )
