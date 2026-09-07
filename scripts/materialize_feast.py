"""Materializes Feast's feature view for exactly the dataset's own date range.

    python -m scripts.build_feast_offline_source
    cd feature_repo && ../.venv/Scripts/feast apply && cd ..
    python -m scripts.materialize_feast

The synthetic dataset is dated relative to when it was generated
(``src/data/historical_generator.py``'s ``start_date``), which has nothing
to do with the wall-clock date someone actually runs this on. ``feast
materialize-incremental <now>`` anchors to the real "now" and, if that now is
far from the dataset's own dates, incrementally materializes an empty
window. Reading the dataset's actual min/max timestamps and materializing
exactly that range sidesteps the mismatch entirely.
"""

from pathlib import Path

import pandas as pd
from feast import FeatureStore


def main() -> None:
    events = pd.read_parquet("data/historical_events.parquet")
    start = pd.Timestamp(events["event_time"].min()).to_pydatetime()
    end = pd.Timestamp(events["event_time"].max()).to_pydatetime() + pd.Timedelta(days=1)

    store = FeatureStore(repo_path=str(Path(__file__).parent.parent / "feature_repo"))
    print(f"Materializing user_daily_stats from {start} to {end}")
    store.materialize(start_date=start, end_date=end)
    print("Done. Try `python -m scripts.demo_feast_point_in_time` next.")


if __name__ == "__main__":
    main()
