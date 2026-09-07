"""Appends every decision to a JSON-Lines file — the system's audit trail.

One line per decision, so it can be tailed live, replayed, or loaded into
pandas with ``pd.read_json(path, lines=True)`` for the kind of "how many
BLOCKs did we issue last hour and why" question an operator actually asks.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from src.decisions.policy import DecisionRecord

DEFAULT_LOG_PATH = Path("data/decision_log.jsonl")


class DecisionLogger:
    def __init__(self, path: Path = DEFAULT_LOG_PATH):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def log(self, record: DecisionRecord) -> None:
        line = json.dumps(record.to_dict())
        with self._lock:
            with open(self.path, "a") as f:
                f.write(line + "\n")
