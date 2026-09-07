from src.data.historical_generator import HistoricalDataGenerator
from src.models.train import chronological_split, evaluate, pick_threshold, train_and_evaluate
from src.features.temporal_features import FEATURE_COLUMNS, compute_features
import numpy as np


def _small_dataset():
    # A small, fast population — just enough days/users/fraud rate to
    # guarantee a handful of positives without the multi-second cost of the
    # full 1,500-user dataset used for the real training run.
    generator = HistoricalDataGenerator(seed=1, n_days=20, fraud_rate=0.5)
    return generator.generate(n_users=40)


def test_chronological_split_respects_time_order():
    df = _small_dataset()
    features = compute_features(df)

    train, val, test, bounds = chronological_split(features, train_frac=0.6, val_frac=0.2)

    assert len(train) + len(val) + len(test) == len(features)
    if len(train) and len(test):
        assert train["event_time"].max() <= bounds.train_end
        assert test["event_time"].min() >= bounds.val_end


def test_pick_threshold_is_between_zero_and_one():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.3, 0.6, 0.8, 0.9])

    threshold = pick_threshold(y_true, scores)
    assert 0.0 <= threshold <= 1.0


def test_pick_threshold_handles_no_positives_gracefully():
    y_true = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.3, 0.4])
    # Must not raise, even though F1 is undefined everywhere.
    threshold = pick_threshold(y_true, scores)
    assert isinstance(threshold, float)


def test_evaluate_returns_expected_keys():
    y_true = np.array([0, 0, 1, 1])
    scores = np.array([0.1, 0.4, 0.6, 0.9])

    metrics = evaluate(y_true, scores, threshold=0.5)

    for key in ("threshold", "roc_auc", "pr_auc", "precision", "recall", "f1", "confusion_matrix"):
        assert key in metrics
    assert metrics["confusion_matrix"]["tp"] == 2


def test_train_and_evaluate_end_to_end_smoke():
    df = _small_dataset()

    model, metadata = train_and_evaluate(df, train_frac=0.6, val_frac=0.2)

    assert hasattr(model, "predict_proba")
    assert metadata["feature_columns"] == FEATURE_COLUMNS
    assert 0.0 <= metadata["threshold"] <= 1.0
    assert metadata["split"]["n_train"] + metadata["split"]["n_val"] + metadata["split"]["n_test"] > 0
    test_metrics = metadata["test_metrics"]
    assert 0.0 <= test_metrics["precision"] <= 1.0
    assert 0.0 <= test_metrics["recall"] <= 1.0
