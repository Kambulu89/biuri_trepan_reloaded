"""Testes básicos do subsistema de contrafactuais."""
import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from counterfactuals._bootstrap import setup

setup()

from counterfactuals.improve_surrogate import improve_surrogate, evaluate_improvement
from core.trepan_original import TrepanOriginalExtractor


@pytest.fixture
def toy_setup():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=0)
    mlp.fit(X, y)
    mlp.predict = mlp.predict
    mlp.predict_proba = mlp.predict_proba

    extractor = TrepanOriginalExtractor()
    extractor.extract_tree(
        mlp, X, y, sample_size=200,
        feature_names=[f"f{i}" for i in range(4)],
        class_names=["0", "1"],
    )
    original = extractor.explainer_tree

    cfs = []
    for i in range(5):
        cf = X[i].copy()
        cf[0] *= -1
        cfs.append({"cf": cf, "original_class": int(y[i])})

    return mlp, X, y, extractor, original, cfs


def test_improve_surrogate_returns_tree(toy_setup):
    mlp, X, y, extractor, original, cfs = toy_setup
    extractor.explainer_tree = original
    improved, stats = improve_surrogate(
        mlp_model=mlp,
        X_train=X,
        y_train=y,
        tree_extractor=extractor,
        cf_list=cfs,
        feature_names=[f"f{i}" for i in range(4)],
        class_names=["0", "1"],
        max_cfs=3,
        seed=0,
    )
    assert improved is not None
    assert stats["total_cfs"] == 5


def test_evaluate_improvement_metrics(toy_setup):
    mlp, X, y, extractor, original, cfs = toy_setup
    extractor.explainer_tree = original
    improved, _ = improve_surrogate(
        mlp_model=mlp,
        X_train=X,
        y_train=y,
        tree_extractor=extractor,
        cf_list=cfs,
        feature_names=[f"f{i}" for i in range(4)],
        class_names=["0", "1"],
        max_cfs=3,
        seed=0,
    )
    metrics = evaluate_improvement(mlp, X, y, original, improved)
    assert "fidelity_improved" in metrics
    assert "accuracy_improved" in metrics
