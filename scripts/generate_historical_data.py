"""Generate the reproducible historical event dataset used for training.

    python -m scripts.generate_historical_data
"""

from pathlib import Path

from src.data.historical_generator import HistoricalDataGenerator


def main():
    # A "realistic" sub-1% account-attack rate, at a portfolio-scale
    # population, leaves too few positive examples for a stable chronological
    # split (500 users at 4.5% gave 79 positive transactions total). The
    # fraud rate here is deliberately dialled up so there is enough signal to
    # train and evaluate on — see docs/understanding.md for the trade-off.
    generator = HistoricalDataGenerator(seed=42, n_days=60, fraud_rate=0.08)

    df = generator.generate(n_users=1500)

    output_path = Path("data/historical_events.parquet")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    n_txn = int((df["event_type"] == "transaction").sum())
    n_fraud_txn = int(((df["event_type"] == "transaction") & df["is_fraud_episode"]).sum())

    print(f"Generated {len(df):,} events across {df['user_id'].nunique():,} users")
    print(f"  date range: {df['event_time'].min()} -> {df['event_time'].max()}")
    print(f"Saved to: {output_path}")
    print()
    print("Event type distribution:")
    print(df["event_type"].value_counts())
    print()
    print(f"Transactions: {n_txn:,}  |  fraud-episode transactions: {n_fraud_txn:,} "
          f"({100 * n_fraud_txn / max(n_txn, 1):.2f}%)")
    print()
    print("Episode type distribution (fraud events only):")
    print(df.loc[df["is_fraud_episode"], "episode_type"].value_counts())


if __name__ == "__main__":
    main()
