import pytest

from src.features.temporal_features import FEATURE_COLUMNS
from src.models.infer import RiskModel


def test_load_raises_clearly_when_no_artifact(tmp_path):
    with pytest.raises(FileNotFoundError):
        RiskModel.load(model_path=tmp_path / "does_not_exist.joblib")


def test_score_returns_a_probability():
    model = RiskModel.load()
    features = {col: 0.0 for col in model.feature_columns}
    score = model.score(features)
    assert 0.0 <= score <= 1.0


def test_score_fills_missing_features_with_zero_rather_than_crashing():
    model = RiskModel.load()
    score = model.score({})  # nothing provided at all
    assert 0.0 <= score <= 1.0


def test_metadata_feature_columns_match_temporal_features_module():
    model = RiskModel.load()
    assert model.feature_columns == FEATURE_COLUMNS


def test_default_threshold_is_between_zero_and_one():
    model = RiskModel.load()
    assert 0.0 <= model.default_threshold <= 1.0
