"""Garante que Trepan Reloaded treina com oráculo MLP, nunca rótulos reais."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trepan_reloaded_extractor import TrepanReloadedExtractor


def test_ensure_training_label_diversity_never_uses_real_labels():
    ext = TrepanReloadedExtractor(ontology=object())

    X = np.random.randn(50, 4)
    y_oracle = np.zeros(50, dtype=int)
    y_real = np.ones(50, dtype=int)

    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=200, random_state=42)
    mlp.fit(X, y_oracle)
    ext._mlp_predict = lambda model, X_in, names=None: np.zeros(len(X_in), dtype=int)

    result = ext._ensure_training_label_diversity(
        y_oracle.copy(), X, X, y_real, mlp, [f"f{i}" for i in range(4)]
    )
    assert np.array_equal(result, y_oracle), "Fallback não deve misturar rótulos reais"


def test_reloaded_original_mode_audit_uses_mlp_oracle():
    rng = np.random.default_rng(0)
    X = rng.random((80, 4))
    y_real = rng.integers(0, 2, size=80)

    mlp = MLPClassifier(max_iter=300, random_state=0)
    mlp.fit(X, y_real)

    X_train, X_test = X[:60], X[60:]
    y_train, y_test = y_real[:60], y_real[60:]

    ext = TrepanReloadedExtractor(ontology=None)
    ext.extract_tree(
        mlp, X, y_real,
        sample_size=400,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    assert ext.explainer_tree is not None
    assert ext.last_audit.get('training_target') == 'MLP Original predictions'
    assert 'trepan_fidelity' in ext.last_audit


def test_fidelity_retry_configs_exist():
    ext = TrepanReloadedExtractor(ontology=None)
    assert ext.FIDELITY_TARGET == 0.90
    assert len(ext.RELOADED_FIDELITY_CONFIGS) >= 2
