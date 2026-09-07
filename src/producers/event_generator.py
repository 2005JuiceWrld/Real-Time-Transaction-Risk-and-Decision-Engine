"""Generates a single *normal* (non-fraud) event for a synthetic user.

Shared by the bulk historical simulation (``src/data/historical_generator.py``)
and the live Kafka producer (``src/producers/kafka_producer.py``), so a
"normal" transaction looks the same whether it was produced for a training
set or streamed live in the demo.
"""

from __future__ import annotations

from datetime import datetime
from uuid import uuid4

import numpy as np

from src.data.user_population import SimUser
from src.producers.event_schema import TransactionEvent


class EventGenerator:
    # A slice of a real user's own history looks a little like an attack
    # every so often, purely by chance — an upgraded phone, a big one-off
    # purchase. Without that, "new device" or "large amount" alone would be
    # a perfect tell rather than one signal among several, which is not how
    # real fraud detection works and makes for a suspiciously easy dataset.
    BENIGN_NEW_DEVICE_RATE = 0.03
    BENIGN_LARGE_PURCHASE_RATE = 0.02

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def generate_amount(self, user: SimUser, *, large: bool = False) -> float:
        loc = user.average_amount * (self.rng.uniform(2.5, 4.0) if large else 1.0)
        amount = self.rng.normal(loc=loc, scale=user.average_amount * user.amount_variability)
        return max(50.0, round(float(amount), 2))

    def generate_transaction(
        self,
        user: SimUser,
        event_time: datetime,
    ) -> TransactionEvent:
        if self.rng.random() < self.BENIGN_NEW_DEVICE_RATE:
            device_id = f"personal_device_{self.rng.integers(1000, 9999)}_{user.user_id}"
        else:
            device_id = user.random_home_device(self.rng)

        large_purchase = self.rng.random() < self.BENIGN_LARGE_PURCHASE_RATE
        amount = self.generate_amount(user, large=large_purchase)

        return TransactionEvent(
            event_id=str(uuid4()),
            user_id=user.user_id,
            event_time=event_time,
            event_type="transaction",
            amount=amount,
            currency="INR",
            device_id=device_id,
            location=user.home_location,
            session_id=f"session_{user.user_id}_{int(event_time.timestamp())}",
            merchant_id=f"merchant_{self.rng.integers(100, 999)}",
            transaction_type="online_purchase",
            authentication_status="success",
        )

    def generate_login(
        self,
        user: SimUser,
        event_time: datetime,
        *,
        force_failure: bool = False,
    ) -> TransactionEvent:
        failed = force_failure or self.rng.random() < user.authentication_failure_rate
        return TransactionEvent(
            event_id=str(uuid4()),
            user_id=user.user_id,
            event_time=event_time,
            event_type="login" if not failed else "authentication_failure",
            device_id=user.random_home_device(self.rng),
            location=user.home_location,
            session_id=f"session_{user.user_id}_{int(event_time.timestamp())}",
            authentication_status="failure" if failed else "success",
        )
