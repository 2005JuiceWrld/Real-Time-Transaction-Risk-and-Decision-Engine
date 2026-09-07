"""Demonstrates point-in-time-correct offline retrieval vs. the online snapshot.

    python -m scripts.build_feast_offline_source
    cd feature_repo && feast apply && cd ..
    python -m scripts.demo_feast_point_in_time

For a handful of real historical transactions, prints:

* the offline, point-in-time-correct feature values Feast reconstructs as
  of each transaction's own moment (what training actually saw), and
* the online store's current snapshot for those same users (what's been
  materialized "as of now") — expected to be *different* numbers, because
  the transactions sampled happened in the past and the online store only
  ever holds the latest materialized day.

That difference is the whole point: point-in-time correctness means asking
"what did we know then" and "what do we know now" are answered by two
different, deliberate queries — never the same cached number reused for both.
"""

from pathlib import Path

import pandas as pd
from feast import FeatureStore

FEATURES = [
    "user_daily_stats:avg_amount_30d",
    "user_daily_stats:txn_count_30d",
    "user_daily_stats:distinct_devices_30d",
]


def main() -> None:
    store = FeatureStore(repo_path=str(Path(__file__).parent.parent / "feature_repo"))

    txns = pd.read_parquet("data/historical_events.parquet")
    txns = txns[txns["event_type"] == "transaction"].sort_values("event_time")
    sample = (
        txns.sample(n=min(5, len(txns)), random_state=1)[["user_id", "event_time"]]
        .rename(columns={"event_time": "event_timestamp"})
        .reset_index(drop=True)
    )

    print("Sampled transactions (entity timestamps for the point-in-time join):")
    print(sample)
    print()

    historical = store.get_historical_features(entity_df=sample, features=FEATURES).to_df()
    print("Offline, point-in-time-correct features (as of each transaction's own moment):")
    print(historical)
    print()

    online = store.get_online_features(
        features=FEATURES,
        entity_rows=[{"user_id": uid} for uid in sample["user_id"]],
    ).to_df()
    print("Online features (the latest materialized snapshot, i.e. 'right now'):")
    print(online)


if __name__ == "__main__":
    main()
