"""Feast's one feature view: a slow-moving, daily-grained user baseline.

This is deliberately a *second*, complementary feature system, not a
replacement for ``src/features/temporal_features.py``. The model's actual
inputs (velocity, recency, z-scores against a rolling baseline) need to be
fresh to the second, which is exactly what Feast's periodic materialization
is not built for — so those stay in the synchronous Redis path
(``src/features/feature_store_client.py``). What Feast is genuinely good
at, and what it demonstrates here, is a *point-in-time-correct* join between
a slow-changing offline table and a set of event timestamps, plus serving
the latest materialized snapshot of that same table online.

    python -m scripts.build_feast_offline_source   # writes the parquet source
    cd feature_repo && feast apply
    feast materialize-incremental $(date -u +%Y-%m-%dT%H:%M:%S)
    python -m scripts.demo_feast_point_in_time
"""

from datetime import timedelta

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64, Int64
from feast.value_type import ValueType

user = Entity(name="user_id", join_keys=["user_id"], value_type=ValueType.STRING)

user_daily_stats_source = FileSource(
    name="user_daily_stats_source",
    path="../data/user_daily_stats.parquet",
    timestamp_field="event_timestamp",
)

user_daily_stats_fv = FeatureView(
    name="user_daily_stats",
    entities=[user],
    ttl=timedelta(days=35),
    schema=[
        Field(name="avg_amount_30d", dtype=Float64),
        Field(name="txn_count_30d", dtype=Int64),
        Field(name="distinct_devices_30d", dtype=Int64),
    ],
    online=True,
    source=user_daily_stats_source,
)
