"""TREPAN Original deve imitar o MLP Original, nunca aprender y real directamente."""
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

from core.trepan_original import TrepanOriginalClassifier, TrepanOriginalExtractor


def test_trepan_trains_on_mlp_oracle_not_real_labels():
    rng = np.random.default_rng(0)
    X = rng.random((100, 4))
    y_real = ((X[:, 0] + X[:, 1]) > 1.0).astype(int)

    mlp = MLPClassifier(hidden_layer_sizes=(12,), max_iter=800, random_state=0)
    mlp.fit(X[:75], y_real[:75])

    X_train, X_test = X[:75], X[75:]
    y_train, y_test = y_real[:75], y_real[75:]

    extractor = TrepanOriginalExtractor()
    extractor.extract_tree(
        mlp, X, y_real,
        sample_size=500,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )

    tree = extractor.explainer_tree
    assert tree is not None
    y_mlp = mlp.predict(X_test)
    y_tree = tree.predict(X_test)
    fidelity = float(np.mean(y_mlp == y_tree))
    assert fidelity >= 0.70
    assert extractor.last_audit['training_target'] == 'MLP Original predictions'
    assert extractor.last_audit['trepan_fidelity'] == pytest.approx(fidelity, abs=1e-6)


def test_trepan_extractor_passes_oracle_not_real_target(monkeypatch):
    rng = np.random.default_rng(1)
    X = rng.random((60, 3))
    y_real = (X[:, 0] > 0.5).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=600, random_state=0).fit(X, y_real)

    captured = {}
    original_fit = TrepanOriginalClassifier.fit

    def spy_fit(self, X, y=None, *, oracle=None, sample_weight=None, feature_names=None):
        captured['y'] = y
        captured['oracle'] = oracle
        return original_fit(
            self, X, y=y, oracle=oracle,
            sample_weight=sample_weight, feature_names=feature_names,
        )

    monkeypatch.setattr(TrepanOriginalClassifier, 'fit', spy_fit)
    extractor = TrepanOriginalExtractor()
    extractor.extract_tree(
        mlp, X, np.ones_like(y_real),
        sample_size=300,
        X_train=X,
        y_train=np.ones_like(y_real),
    )

    assert captured['oracle'] is mlp
    assert captured['y'] is None
    assert extractor.last_audit['training_target'] == 'MLP Original predictions'
