"""Train the transaction risk model.

    python -m scripts.train_model

Splits chronologically (never randomly — a model that peeked at "the future"
relative to its own training cutoff would report metrics no live system
could ever reproduce), trains an XGBoost classifier, picks an operating
threshold on the validation set, reports test-set metrics, and saves a
reproducible model artifact plus its metadata.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.features.temporal_features import FEATURE_COLUMNS, LABEL_COLUMN, compute_features

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "risk_model.joblib"
METADATA_PATH = MODEL_DIR / "risk_model_metadata.json"

RANDOM_SEED = 42


@dataclass
class SplitBounds:
    train_end: pd.Timestamp
    val_end: pd.Timestamp


def chronological_split(
    features: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, SplitBounds]:
    """Split by *time*, not by row shuffling or by user.

    The cut points are chosen so that ``train_frac`` of the time range (not
    of the row count) falls in train, etc. — events don't arrive at a
    constant rate, but the split still has to represent "everything the
    model would have known as of a given moment in production".
    """

    features = features.sort_values("event_time").reset_index(drop=True)
    t0, t1 = features["event_time"].min(), features["event_time"].max()
    span = t1 - t0

    train_end = t0 + span * train_frac
    val_end = t0 + span * (train_frac + val_frac)

    train = features[features["event_time"] < train_end]
    val = features[(features["event_time"] >= train_end) & (features["event_time"] < val_end)]
    test = features[features["event_time"] >= val_end]

    return train, val, test, SplitBounds(train_end=train_end, val_end=val_end)


def pick_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    """The score threshold maximising F1 on a held-out (validation) set.

    Chosen on validation, applied to test — never fit and picked on the same
    split, which is the same chronological-leakage mistake at the threshold
    level rather than the feature level.
    """

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    # precision_recall_curve returns one more point than thresholds (for
    # recall=0); align by dropping that last point.
    precision, recall = precision[:-1], recall[:-1]
    f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall + 1e-12), 0.0)
    if len(f1) == 0:
        return 0.5
    return float(thresholds[int(np.argmax(f1))])


def evaluate(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    preds = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()

    return {
        "threshold": threshold,
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "pr_auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else float("nan"),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "n_samples": int(len(y_true)),
        "n_positive": int(y_true.sum()),
    }


def train_and_evaluate(
    events: pd.DataFrame,
    train_frac: float = 0.60,
    val_frac: float = 0.20,
) -> tuple[xgb.XGBClassifier, dict]:
    features = compute_features(events)
    train, val, test, bounds = chronological_split(features, train_frac, val_frac)

    x_train, y_train = train[FEATURE_COLUMNS], train[LABEL_COLUMN].astype(int)
    x_val, y_val = val[FEATURE_COLUMNS], val[LABEL_COLUMN].astype(int)
    x_test, y_test = test[FEATURE_COLUMNS], test[LABEL_COLUMN].astype(int)

    n_pos = max(int(y_train.sum()), 1)
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)

    val_scores = model.predict_proba(x_val)[:, 1]
    threshold = pick_threshold(y_val.to_numpy(), val_scores)

    test_scores = model.predict_proba(x_test)[:, 1]
    test_metrics = evaluate(y_test.to_numpy(), test_scores, threshold)
    val_metrics = evaluate(y_val.to_numpy(), val_scores, threshold)

    metadata = {
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "seed": RANDOM_SEED,
        "feature_columns": FEATURE_COLUMNS,
        "label_column": LABEL_COLUMN,
        "split": {
            "train_frac": train_frac,
            "val_frac": val_frac,
            "train_end": bounds.train_end.isoformat(),
            "val_end": bounds.val_end.isoformat(),
            "n_train": len(train),
            "n_val": len(val),
            "n_test": len(test),
        },
        "threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "model_params": model.get_params(),
        "versions": {
            "python": platform.python_version(),
            "xgboost": xgb.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }

    return model, metadata


def save_artifact(model: xgb.XGBClassifier, metadata: dict) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def _print_report(metadata: dict) -> None:
    test = metadata["test_metrics"]
    print("=" * 60)
    print("Chronological split")
    print(f"  train: {metadata['split']['n_train']:,} rows  (< {metadata['split']['train_end']})")
    print(f"  val:   {metadata['split']['n_val']:,} rows")
    print(f"  test:  {metadata['split']['n_test']:,} rows  (>= {metadata['split']['val_end']})")
    print()
    print(f"Operating threshold (chosen on validation): {metadata['threshold']:.4f}")
    print()
    print("Test-set metrics")
    print(f"  ROC-AUC:   {test['roc_auc']:.4f}")
    print(f"  PR-AUC:    {test['pr_auc']:.4f}")
    print(f"  Precision: {test['precision']:.4f}")
    print(f"  Recall:    {test['recall']:.4f}")
    print(f"  F1:        {test['f1']:.4f}")
    cm = test["confusion_matrix"]
    print(f"  Confusion matrix: TN={cm['tn']} FP={cm['fp']} FN={cm['fn']} TP={cm['tp']}")
    print(f"  Positives in test set: {test['n_positive']} / {test['n_samples']}")
    print("=" * 60)


def main() -> None:
    events = pd.read_parquet("data/historical_events.parquet")
    model, metadata = train_and_evaluate(events)
    save_artifact(model, metadata)
    _print_report(metadata)
    print(f"\nSaved model to {MODEL_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
