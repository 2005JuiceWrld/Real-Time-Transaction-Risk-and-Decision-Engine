"""Streams the historical dataset to Kafka as if it were happening live.

    python -m src.producers.kafka_producer --speed 600 --late-rate 0.03

Rather than re-simulating a live population from scratch, this replays
``data/historical_events.parquet``: the same realistic per-user behaviour and
fraud episodes already validated by training, just paced out in wall-clock
time instead of arriving all at once. ``--speed`` compresses simulated time
(600 means one simulated hour arrives roughly every 6 real seconds).

A tunable fraction of events are deliberately held back and sent late
(``--late-rate``, ``--late-delay-seconds``) — sent to Kafka out of their
original event-time order — so Flink's watermark and late-data handling
(``src/streaming/flink_job.py``) has real out-of-order data to demonstrate
against, rather than a stream that is trivially already sorted.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from kafka import KafkaProducer

TOPIC = "transactions"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("kafka_producer")


def _load_events(limit: int | None) -> pd.DataFrame:
    df = pd.read_parquet("data/historical_events.parquet")
    df["event_time"] = pd.to_datetime(df["event_time"])
    df = df.sort_values("event_time").reset_index(drop=True)
    if limit:
        df = df.head(limit)
    return df


def _serialize(row: pd.Series) -> dict:
    # pandas represents a missing value in an object-dtype column (amount,
    # merchant_id, ... on a login/auth-failure row) as float NaN, not None.
    # json.dumps happily emits the non-standard `NaN` token for it, and
    # Pydantic's `ge=0` constraint then rejects it outright (NaN compares
    # false to everything) — every non-transaction event was being dropped
    # as "malformed" downstream until this replaced NaN with a real null.
    row = row.where(pd.notna(row), None)
    payload = row.to_dict()
    # Drop simulation-only ground truth — a live producer would never know
    # this, and no downstream consumer is allowed to use it as a feature.
    payload.pop("is_fraud_episode", None)
    payload.pop("episode_type", None)
    payload["event_time"] = pd.Timestamp(payload["event_time"]).isoformat()
    payload["produced_at"] = datetime.now(timezone.utc).isoformat()
    return payload


def replay(
    bootstrap_servers: str,
    speed: float,
    late_rate: float,
    late_delay_range: tuple[float, float],
    limit: int | None,
    loop: bool,
    seed: int,
) -> None:
    df = _load_events(limit)
    rng = np.random.default_rng(seed)

    producer = KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8"),
        linger_ms=20,
    )
    logger.info("connected to Kafka at %s, replaying %d events at %.0fx speed", bootstrap_servers, len(df), speed)

    passes = 0
    while True:
        _replay_once(producer, df, rng, speed, late_rate, late_delay_range)
        passes += 1
        logger.info("completed replay pass %d", passes)
        if not loop:
            break

    producer.flush()


def _replay_once(producer, df, rng, speed, late_rate, late_delay_range) -> None:
    sim_start = df["event_time"].iloc[0]
    wall_start = time.monotonic()
    pending_late: list[tuple[float, dict]] = []
    sent = 0

    for _, row in df.iterrows():
        sim_elapsed = (row["event_time"] - sim_start).total_seconds()
        target_wall = wall_start + sim_elapsed / speed

        sleep_for = target_wall - time.monotonic()
        if sleep_for > 0:
            time.sleep(min(sleep_for, 2.0))

        payload = _serialize(row)

        if rng.random() < late_rate:
            delay = float(rng.uniform(*late_delay_range))
            pending_late.append((time.monotonic() + delay, payload))
        else:
            producer.send(TOPIC, key=row["user_id"], value=payload)
            sent += 1

        pending_late = _flush_due(producer, pending_late)

        if sent and sent % 2000 == 0:
            logger.info("sent %d events (%d held for late delivery)", sent, len(pending_late))

    # Anything still pending at the end of the pass goes out now, late by
    # definition but not lost.
    for _due, payload in pending_late:
        producer.send(TOPIC, key=payload["user_id"], value=payload)
    producer.flush()


def _flush_due(producer, pending_late: list[tuple[float, dict]]) -> list[tuple[float, dict]]:
    now = time.monotonic()
    still_pending = []
    for due, payload in pending_late:
        if due <= now:
            producer.send(TOPIC, key=payload["user_id"], value=payload)
        else:
            still_pending.append((due, payload))
    return still_pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap-servers", default=os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"))
    parser.add_argument("--speed", type=float, default=600.0, help="simulated-seconds per real-second")
    parser.add_argument("--late-rate", type=float, default=0.03, help="fraction of events sent out of order")
    parser.add_argument("--late-delay-min", type=float, default=2.0)
    parser.add_argument("--late-delay-max", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=None, help="only replay the first N events")
    parser.add_argument("--loop", action="store_true", help="replay the dataset repeatedly")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    replay(
        bootstrap_servers=args.bootstrap_servers,
        speed=args.speed,
        late_rate=args.late_rate,
        late_delay_range=(args.late_delay_min, args.late_delay_max),
        limit=args.limit,
        loop=args.loop,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
