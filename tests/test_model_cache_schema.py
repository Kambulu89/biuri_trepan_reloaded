"""Testes de invalidação de cache por schema de features."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_cache import validate_cached_training


def test_cache_invalidated_on_feature_count_mismatch(capsys):
    cached = {
        "n_features": 3,
        "feature_space": "original",
        "feature_names": ["a", "b", "c"],
    }
    ok = validate_cached_training(
        cached,
        expected_n_features=5,
        expected_feature_space="original",
        feature_names=["a", "b", "c", "d", "e"],
    )
    assert ok is False
    out = capsys.readouterr().out
    assert "[CACHE INVALIDATED]" in out
    assert "expected 3 features" in out or "current matrix has 5" in out


def test_cache_valid_when_schema_matches():
    cached = {
        "n_features": 3,
        "feature_space": "original",
        "feature_names": ["a", "b", "c"],
    }
    assert validate_cached_training(
        cached,
        expected_n_features=3,
        expected_feature_space="original",
        feature_names=["a", "b", "c"],
    )
