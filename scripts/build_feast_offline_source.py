"""Builds the daily per-user baseline table Feast's offline source reads.

    python -m scripts.build_feast_offline_source

One row per (user, day): the 30-day trailing average amount, transaction
count, and distinct device count *as of the start of that day* — computed
from transactions strictly before midnight of that day, never including the
day's own transactions. That is what makes the later point-in-time join
honest: when ``get_historical_features`` looks up "what did we know about
this user as of their 2pm transaction on day D", it should get the same
answer whether that transaction happened at 2am or 11pm on day D — and it
does, because both resolve to day D's row, computed only from days before D.
"""

from pathlib import Path

import pandas as pd

OUTPUT_PATH = Path("data/user_daily_stats.parquet")


def main() -> None:
    events = pd.read_parquet("data/historical_events.parquet")
    events["event_time"] = pd.to_datetime(events["event_time"])
    txns = events[events["event_type"] == "transaction"].copy()
    txns["day"] = txns["event_time"].dt.floor("D")

    rows = []
    for user_id, group in txns.groupby("user_id"):
        group = group.sort_values("event_time")
        days = pd.date_range(pd.Timestamp(group["day"].min()), pd.Timestamp(group["day"].max()), freq="D")

        for day in days:
            window_start = day - pd.Timedelta(days=30)
            window = group[(group["event_time"] >= window_start) & (group["event_time"] < day)]

            rows.append(
                {
                    "user_id": user_id,
                    "event_timestamp": day,
                    "avg_amount_30d": float(window["amount"].mean()) if len(window) else 0.0,
                    "txn_count_30d": int(len(window)),
                    "distinct_devices_30d": int(window["device_id"].nunique()) if len(window) else 0,
                }
            )

    out = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {len(out):,} user-day rows for {out['user_id'].nunique():,} users to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
