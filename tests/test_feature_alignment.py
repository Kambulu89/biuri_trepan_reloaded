"""Testes do Feature Alignment Engine."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.feature_alignment import align_feature_spaces, oracle_n_features
from core.metrics_comparator import MetricsComparator


@pytest.fixture
def enriched_matrix():
    rng = np.random.default_rng(0)
    base_names = ["age", "salary", "city"]
    full_names = base_names + [f"onto_extra_{i}" for i in range(21)]
    X_base = rng.random((20, 3))
    X_full = np.hstack([X_base, rng.random((20, 21))])
    return X_full, full_names, base_names


def test_direct_alignment_same_width(enriched_matrix):
    X_full, full_names, _ = enriched_matrix
    mlp = MLPClassifier(max_iter=100, random_state=0)
    mlp.fit(X_full, np.zeros(20))
    aligned = align_feature_spaces(mlp, X_full, feature_names=full_names, log=False)
    assert aligned.shape == X_full.shape


def test_project_original_oracle_from_enriched_matrix(enriched_matrix):
    X_full, full_names, base_names = enriched_matrix
    mlp_orig = MLPClassifier(max_iter=100, random_state=0)
    mlp_orig.fit(X_full[:, :3], np.zeros(20))
    aligned = align_feature_spaces(
        mlp_orig,
        X_full,
        feature_names=full_names,
        original_feature_names=base_names,
        oracle_type="MLP Original",
        log=False,
    )
    assert aligned.shape == (20, 3)


def test_missing_names_fails_fast_without_positional_fallback():
    rng = np.random.default_rng(1)
    X = rng.random((15, 24))
    mlp = MLPClassifier(max_iter=100, random_state=0)
    mlp.fit(X[:, :3], np.zeros(15))
    with pytest.raises(ValueError, match="feature_names explícitos"):
        align_feature_spaces(mlp, X, log=False)


def test_mlp_oracle_matrix_no_longer_raises(enriched_matrix):
    X_full, full_names, base_names = enriched_matrix
    mlp_orig = MLPClassifier(max_iter=100, random_state=0)
    mlp_orig.fit(X_full[:, :3], np.zeros(20))
    mc = MetricsComparator(enable_cache=False)
    X_oracle = mc._mlp_oracle_matrix(
        mlp_orig,
        X_full,
        matrix_feature_names=full_names,
        oracle_feature_names=base_names,
    )
    assert X_oracle.shape == (20, 3)


def test_reloaded_fidelity_reference_prefers_onto_for_enriched():
    rng = np.random.default_rng(2)
    X_aug = rng.random((10, 24))
    mlp_onto = MLPClassifier(max_iter=100, random_state=0)
    mlp_onto.fit(X_aug, np.zeros(10))
    names = [f"f{i}" for i in range(24)]
    ref = MetricsComparator._reloaded_fidelity_reference(
        mlp_onto, names, n_matrix_features=24
    )
    assert ref == "mlp_onto"


def test_pipeline_oracle_n_features():
    rng = np.random.default_rng(3)
    X = rng.random((12, 5))
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=100, random_state=0)),
    ])
    pipe.fit(X, np.zeros(12))
    assert oracle_n_features(pipe) == 5
