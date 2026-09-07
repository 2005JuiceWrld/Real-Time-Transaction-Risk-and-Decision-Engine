"""Consumes the raw transaction stream and turns every transaction into a decision.

    python -m src.serving.stream_scorer

This is the "ML Inference -> Decision Engine -> ALLOW/REVIEW/BLOCK" box in
the architecture diagram, triggered by a live Kafka message instead of an
HTTP request. It shares every piece of logic with the FastAPI path
(``src/serving/api.py``) — same feature store, same model, same policy — so
a transaction gets the same decision whichever way it entered the system.

Runs its own small Prometheus endpoint (default :8001) because it is a
separate process from the API server and cannot share its metrics registry.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time

import redis
from kafka import KafkaConsumer
from prometheus_client import start_http_server

from src.decisions.decision_log import DecisionLogger
from src.decisions.policy import DecisionEngine
from src.features.feature_store_client import FeatureStoreClient
from src.models.infer import RiskModel
from src.producers.event_schema import TransactionEvent
from src.serving.metrics import (
    DECISIONS_TOTAL,
    EVENTS_PROCESSED_TOTAL,
    FEATURE_STORE_ERRORS_TOTAL,
    INFERENCE_LATENCY_SECONDS,
    RISK_SCORE,
)

SOURCE = "stream"
TOPIC = "transactions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("stream_scorer")


def run(bootstrap_servers: str, redis_url: str, metrics_port: int) -> None:
    model = RiskModel.load()
    engine = DecisionEngine.from_yaml(model_version=model.metadata.get("trained_at"))
    decision_logger = DecisionLogger()
    store = FeatureStoreClient(redis.Redis.from_url(redis_url, decode_responses=True))

    consumer = KafkaConsumer(
        TOPIC,
        bootstrap_servers=bootstrap_servers,
        group_id="risk-scoring",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k else None,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    start_http_server(metrics_port)
    logger.info("stream_scorer listening on %s, metrics on :%d", TOPIC, metrics_port)

    for message in consumer:
        try:
            _handle(message.value, model, engine, decision_logger, store)
        except Exception:
            logger.exception("failed to process message offset=%s", message.offset)


def _handle(raw: dict, model: RiskModel, engine: DecisionEngine, decision_logger: DecisionLogger, store: FeatureStoreClient) -> None:
    try:
        event = TransactionEvent.model_validate(raw)
    except Exception:
        logger.warning("dropping malformed message: %s", raw)
        return

    EVENTS_PROCESSED_TOTAL.labels(event_type=event.event_type, source=SOURCE).inc()

    if event.event_type == "transaction" and event.amount is not None:
        start = time.perf_counter()
        try:
            features = store.get_features(
                user_id=event.user_id,
                event_time=event.event_time,
                amount=event.amount,
                device_id=event.device_id,
                location=event.location,
            )
        except Exception:
            FEATURE_STORE_ERRORS_TOTAL.labels(source=SOURCE).inc()
            raise

        risk_score = model.score(features)
        record = engine.decide(
            event_id=event.event_id,
            user_id=event.user_id,
            risk_score=risk_score,
            features=features,
        )
        decision_logger.log(record)

        INFERENCE_LATENCY_SECONDS.labels(source=SOURCE).observe(time.perf_counter() - start)
        RISK_SCORE.labels(source=SOURCE).observe(risk_score)
        DECISIONS_TOTAL.labels(decision=record.decision, source=SOURCE).inc()

        if record.decision != "ALLOW":
            logger.info(
                "%s user=%s risk=%.3f reasons=%s",
                record.decision, event.user_id, risk_score, ",".join(record.reason_codes),
            )

    # Every event — transaction or not — becomes prior history for the
    # user's next one, whether or not it was scored.
    store.record_event(
        user_id=event.user_id,
        event_time=event.event_time,
        event_type=event.event_type,
        amount=event.amount,
        device_id=event.device_id,
        location=event.location,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--redis-url", default=os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    parser.add_argument("--metrics-port", type=int, default=int(os.environ.get("METRICS_PORT", 8001)))
    args = parser.parse_args()

    run(args.bootstrap_servers, args.redis_url, args.metrics_port)


if __name__ == "__main__":
    main()
