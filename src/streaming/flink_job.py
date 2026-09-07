"""PyFlink job: genuine event-time processing over the raw transaction stream.

    docker compose exec flink-jobmanager \\
        flink run -py /opt/flink-job/flink_job.py -pyFiles /opt/flink-job

Runs *inside* the Flink cluster (see ``docker/flink/Dockerfile`` and the
``flink-jobmanager``/``flink-taskmanager`` services in ``docker-compose.yml``)
rather than in this repo's own virtualenv — PyFlink ships its own bundled
JVM/connector wiring that only makes sense alongside a running Flink
cluster.

What this demonstrates, concretely:

* **Event time, not processing time** — windows are keyed off each event's
  own ``event_time`` field, not by when Flink happened to read it off Kafka.
* **Watermarks** — ``for_bounded_out_of_orderness`` tells Flink to wait up to
  30 seconds of event time for stragglers before closing a window, which is
  what actually lets the next point matter:
* **Out-of-order / late events** — the producer (``src/producers/kafka_producer.py``)
  deliberately delivers a few percent of events late; those that arrive
  within the 30-second watermark slack still land in the correct window,
  and any that arrive later than that are routed to a side output instead of
  silently dropped or (worse) corrupting an already-closed window.
* **Rolling windows** — a 1-minute tumbling window per user, aggregated into
  event/transaction counts and amount stats.

This job intentionally does *not* sit on the path a live decision is made
on. That path (``src/serving/api.py`` / ``src/serving/stream_scorer.py``)
keeps its own synchronous, per-event state in Redis (the ``rt:*`` keys) via
``src/features/feature_store_client.py``, because a real-time decision
cannot wait on a window to close. This job is the complementary "speed
layer": periodic, windowed, slightly-delayed-by-design aggregates — the kind
of view a monitoring dashboard or a nightly batch job wants, not the kind an
inference call can block on. See docs/understanding.md for the full
reasoning.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from pyflink.common import Duration, Types
from pyflink.common.watermark_strategy import TimestampAssigner, WatermarkStrategy
from pyflink.datastream import OutputTag, StreamExecutionEnvironment
from pyflink.datastream.connectors.kafka import KafkaOffsetsInitializer, KafkaSource
from pyflink.datastream.functions import ProcessWindowFunction, RichMapFunction
from pyflink.datastream.window import TumblingEventTimeWindows, Time

TOPIC = os.environ.get("KAFKA_TOPIC", "transactions")
WATERMARK_SLACK_SECONDS = 30
WINDOW_MINUTES = 1

LATE_EVENTS_TAG = OutputTag("late-events", Types.PICKLED_BYTE_ARRAY())


def _parse(raw: str) -> dict | None:
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return None


class EventTimeAssigner(TimestampAssigner):
    """Pulls the watermark clock from the event's own ``event_time`` field."""

    def extract_timestamp(self, value: dict, record_timestamp: int) -> int:
        try:
            return int(datetime.fromisoformat(value["event_time"]).timestamp() * 1000)
        except (KeyError, ValueError):
            return record_timestamp


class UserWindowAggregator(ProcessWindowFunction):
    """One record per (user, 1-minute window): what that user did in it."""

    def process(self, key, context, elements):
        events = list(elements)
        txns = [e for e in events if e.get("event_type") == "transaction" and e.get("amount") is not None]
        amounts = [float(e["amount"]) for e in txns]
        auth_failures = sum(1 for e in events if e.get("event_type") == "authentication_failure")

        yield {
            "user_id": key,
            "window_start_ms": context.window().start,
            "window_end_ms": context.window().end,
            "event_count": len(events),
            "txn_count": len(txns),
            "txn_amount_sum": round(sum(amounts), 2),
            "txn_amount_avg": round(sum(amounts) / len(amounts), 2) if amounts else 0.0,
            "distinct_devices": len({e.get("device_id") for e in events if e.get("device_id")}),
            "auth_failure_count": auth_failures,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }


class RedisAggregateSink(RichMapFunction):
    """Writes each window's aggregate into Redis under the ``agg:*`` namespace.

    A `RichFunction`'s ``open()``/``close()`` are exactly the lifecycle hooks
    meant for a network client like this: one Redis connection per parallel
    subtask, opened once, reused for every record, rather than reconnecting
    per event.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 2 * 3600):
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        self.redis = None

    def open(self, runtime_context) -> None:
        import redis

        self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)

    def close(self) -> None:
        if self.redis is not None:
            self.redis.close()

    def map(self, value: dict) -> dict:
        key = f"agg:{value['user_id']}:{value['window_end_ms']}"
        pipe = self.redis.pipeline()
        pipe.hset(key, mapping={k: str(v) for k, v in value.items()})
        pipe.expire(key, self.ttl_seconds)
        pipe.incr("agg:stats:windows_processed")
        pipe.incrby("agg:stats:events_processed", value["event_count"])
        pipe.execute()
        return value


class LateEventCounter(RichMapFunction):
    """Counts events the watermark had already moved past — visible proof
    that "late" is a real, handled case here rather than a theoretical one."""

    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.redis = None

    def open(self, runtime_context) -> None:
        import redis

        self.redis = redis.Redis.from_url(self.redis_url, decode_responses=True)

    def close(self) -> None:
        if self.redis is not None:
            self.redis.close()

    def map(self, value: dict) -> dict:
        self.redis.incr("agg:stats:late_events_dropped")
        return value


def build_job(kafka_bootstrap: str, redis_url: str) -> StreamExecutionEnvironment:
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_parallelism(1)  # a single task is plenty for this data volume, and keeps the demo's output easy to reason about

    source = (
        KafkaSource.builder()
        .set_bootstrap_servers(kafka_bootstrap)
        .set_topics(TOPIC)
        .set_group_id("flink-aggregator")
        .set_starting_offsets(KafkaOffsetsInitializer.latest())
        .set_value_only_deserializer(_string_schema())
        .build()
    )

    watermark_strategy = WatermarkStrategy.for_bounded_out_of_orderness(
        Duration.of_seconds(WATERMARK_SLACK_SECONDS)
    ).with_timestamp_assigner(EventTimeAssigner())

    raw = env.from_source(source, WatermarkStrategy.no_watermarks(), "kafka-transactions")

    parsed = raw.map(_parse, output_type=Types.PICKLED_BYTE_ARRAY()).filter(lambda v: v is not None)

    timestamped = parsed.assign_timestamps_and_watermarks(watermark_strategy)

    windowed = (
        timestamped.key_by(lambda v: v["user_id"], key_type=Types.STRING())
        .window(TumblingEventTimeWindows.of(Time.minutes(WINDOW_MINUTES)))
        .allowed_lateness(Time.seconds(WATERMARK_SLACK_SECONDS))
        .side_output_late_data(LATE_EVENTS_TAG)
        .process(UserWindowAggregator(), output_type=Types.PICKLED_BYTE_ARRAY())
    )

    windowed.map(RedisAggregateSink(redis_url), output_type=Types.PICKLED_BYTE_ARRAY())

    late_events = windowed.get_side_output(LATE_EVENTS_TAG)
    late_events.map(LateEventCounter(redis_url), output_type=Types.PICKLED_BYTE_ARRAY())

    return env


def _string_schema():
    from pyflink.common.serialization import SimpleStringSchema

    return SimpleStringSchema()


def main() -> None:
    kafka_bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    redis_url = os.environ.get("REDIS_URL", "redis://redis:6379/0")

    env = build_job(kafka_bootstrap, redis_url)
    env.execute("transaction-window-aggregator")


if __name__ == "__main__":
    main()
