from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


EventType = Literal[
    "transaction",
    "login",
    "logout",
    "device_added",
    "password_change",
    "authentication_failure",
]


class TransactionEvent(BaseModel):
    event_id: str
    user_id: str
    event_time: datetime
    event_type: EventType

    amount: float | None = Field(default=None, ge=0)
    currency: str = "INR"

    device_id: str
    location: str
    session_id: str

    merchant_id: str | None = None
    transaction_type: str | None = None
    authentication_status: str | None = None