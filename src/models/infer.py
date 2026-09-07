"""The one place a trained model gets turned into a risk score.

Both the Kafka-driven streaming consumer and the FastAPI HTTP endpoint call
through this class rather than touching ``joblib``/XGBoost directly, so
"how a transaction gets scored" has exactly one implementation regardless of
which trigger (a stream message or an HTTP request) caused the scoring.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.features.temporal_features import FEATURE_COLUMNS

DEFAULT_MODEL_PATH = Path("models/risk_model.joblib")
DEFAULT_METADATA_PATH = Path("models/risk_model_metadata.json")


class RiskModel:
    def __init__(self, model, feature_columns: list[str], metadata: dict | None = None):
        self.model = model
        self.feature_columns = feature_columns
        self.metadata = metadata or {}

    @classmethod
    def load(
        cls,
        model_path: Path = DEFAULT_MODEL_PATH,
        metadata_path: Path = DEFAULT_METADATA_PATH,
    ) -> "RiskModel":
        if not model_path.exists():
            raise FileNotFoundError(
                f"No model artifact at {model_path}. Run `python -m scripts.train_model` first."
            )
        model = joblib.load(model_path)

        metadata: dict = {}
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)

        feature_columns = metadata.get("feature_columns", FEATURE_COLUMNS)
        return cls(model=model, feature_columns=feature_columns, metadata=metadata)

    @property
    def default_threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    def score(self, features: dict) -> float:
        """Risk score in [0, 1] for a single transaction's feature dict."""
        row = pd.DataFrame([{col: features.get(col, 0.0) for col in self.feature_columns}])
        return float(self.model.predict_proba(row)[:, 1][0])

    def score_batch(self, features: pd.DataFrame) -> pd.Series:
        """Risk scores for a whole feature table at once (offline evaluation, backtests)."""
        x = features[self.feature_columns]
        return pd.Series(self.model.predict_proba(x)[:, 1], index=features.index)
