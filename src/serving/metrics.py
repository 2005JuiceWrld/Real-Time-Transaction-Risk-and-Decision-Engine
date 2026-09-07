"""Prometheus metric definitions shared by both scoring entry points.

The FastAPI process (``src/serving/api.py``) and the Kafka streaming
consumer (``src/serving/stream_scorer.py``) are separate processes, each
scraped by Prometheus independently — they can't share a registry, but
importing the *definitions* from one place keeps their metric names, label
sets and bucket boundaries identical, so a Grafana panel means the same
thing regardless of which path produced the data.
"""

from prometheus_client import Counter, Histogram

DECISIONS_TOTAL = Counter(
    "decision_engine_decisions_total",
    "Decisions made, by outcome (ALLOW/REVIEW/BLOCK)",
    ["decision", "source"],
)

INFERENCE_LATENCY_SECONDS = Histogram(
    "decision_engine_inference_latency_seconds",
    "Time to fetch features, score, and decide for one transaction",
    ["source"],
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

RISK_SCORE = Histogram(
    "decision_engine_risk_score",
    "Distribution of model risk scores actually produced",
    ["source"],
    buckets=tuple(i / 20 for i in range(21)),
)

EVENTS_PROCESSED_TOTAL = Counter(
    "decision_engine_events_processed_total",
    "Raw events processed, by event type and source",
    ["event_type", "source"],
)

FEATURE_STORE_ERRORS_TOTAL = Counter(
    "decision_engine_feature_store_errors_total",
    "Times the online feature store could not be read or written",
    ["source"],
)
