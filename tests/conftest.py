"""Shared test setup.

The test suite must pass on a fresh checkout with just ``pytest`` — it
should not silently depend on someone having already run
``scripts/train_model.py``. This trains a small, fast model into the real
``models/`` path if nothing is there yet.

Deliberately plain module-level code, *not* a fixture: ``src/serving/api.py``
loads the model artifact at import time (so ``uvicorn src.serving.api:app``
works), and pytest imports every test module during collection — before any
fixture, including a session-scoped autouse one, gets to run. A fixture here
would train the model too late to help ``test_api.py``'s own import. A
conftest.py is imported before the test modules that sit alongside it, so
this runs in time.
"""

from pathlib import Path

from src.data.historical_generator import HistoricalDataGenerator
from src.models.train import save_artifact, train_and_evaluate

MODEL_PATH = Path("models/risk_model.joblib")

if not MODEL_PATH.exists():
    _df = HistoricalDataGenerator(seed=1, n_days=20, fraud_rate=0.5).generate(n_users=40)
    _model, _metadata = train_and_evaluate(_df)
    save_artifact(_model, _metadata)
