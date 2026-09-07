"""The synchronous HTTP entry point into the decision engine.

    uvicorn src.serving.api:app --port 8000

Scores a transaction exactly the way the Kafka streaming consumer
(``src/serving/stream_scorer.py``) does — same feature store, same model,
same policy — just triggered by an HTTP request instead of a Kafka message.
Useful standalone (no Kafka/Flink needed to try it), and it's what the
decision engine actually calls when a transaction needs a synchronous
allow/review/block answer rather than a streamed one.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

import redis
from fastapi import FastAPI
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.decisions.decision_log import DecisionLogger
from src.decisions.policy import DecisionEngine
from src.features.feature_store_client import FeatureStoreClient
from src.models.infer import RiskModel
from src.serving.metrics import (
    DECISIONS_TOTAL,
    EVENTS_PROCESSED_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    RISK_SCORE,
)

SOURCE = "api"


class ScoreRequest(BaseModel):
    event_id: str
    user_id: str
    event_time: datetime
    amount: float = Field(gt=0)
    device_id: str
    location: str


class ScoreResponse(BaseModel):
    event_id: str
    user_id: str
    risk_score: float
    decision: str
    reason_codes: list[str]
    timestamp: str
    model_version: str | None = None


def create_app(redis_client: "redis.Redis | None" = None) -> FastAPI:
    app = FastAPI(
        title="Real-Time Transaction Decision Engine",
        description="Scores a transaction and returns ALLOW / REVIEW / BLOCK.",
        version="0.1.0",
    )

    model = RiskModel.load()
    engine = DecisionEngine.from_yaml(model_version=model.metadata.get("trained_at"))
    logger = DecisionLogger()

    if redis_client is None:
        redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        redis_client = redis.Redis.from_url(redis_url, decode_responses=True)
    store = FeatureStoreClient(redis_client)

    @app.post("/score", response_model=ScoreResponse)
    def score(request: ScoreRequest) -> ScoreResponse:
        start = time.perf_counter()

        features = store.get_features(
            user_id=request.user_id,
            event_time=request.event_time,
            amount=request.amount,
            device_id=request.device_id,
            location=request.location,
        )
        risk_score = model.score(features)
        record = engine.decide(
            event_id=request.event_id,
            user_id=request.user_id,
            risk_score=risk_score,
            features=features,
        )
        logger.log(record)

        # Recorded *after* scoring: this transaction becomes prior history
        # for this user's next request, never for its own.
        store.record_event(
            user_id=request.user_id,
            event_time=request.event_time,
            event_type="transaction",
            amount=request.amount,
            device_id=request.device_id,
            location=request.location,
        )

        INFERENCE_LATENCY_SECONDS.labels(source=SOURCE).observe(time.perf_counter() - start)
        RISK_SCORE.labels(source=SOURCE).observe(risk_score)
        DECISIONS_TOTAL.labels(decision=record.decision, source=SOURCE).inc()
        EVENTS_PROCESSED_TOTAL.labels(event_type="transaction", source=SOURCE).inc()

        return ScoreResponse(**record.to_dict())

    @app.get("/users/{user_id}/features")
    def peek_features(user_id: str, amount: float = 100.0, device_id: str = "probe_device", location: str = "Mumbai") -> dict:
        """Read-only: what the online feature store currently holds for a
        user, as if a transaction with the given amount/device/location
        arrived right now. Does not write anything — purely for demoing and
        debugging what the model would see."""
        return store.get_features(user_id, datetime.now(timezone.utc), amount, device_id, location)

    @app.get("/health")
    def health() -> dict:
        try:
            redis_client.ping()
            redis_ok = True
        except Exception:
            redis_ok = False
        return {
            "status": "ok" if redis_ok else "degraded",
            "redis_connected": redis_ok,
            "model_trained_at": model.metadata.get("trained_at"),
            "decision_threshold": {"allow_below": engine.config.allow_below, "block_at_or_above": engine.config.block_at_or_above},
        }

    @app.get("/metrics")
    def metrics() -> PlainTextResponse:
        return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
