"""Sem ontologia, Trepan-Reloaded deve espelhar exactamente o Trepan-Original."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.metrics_comparator import MetricsComparator
from core.trepan import TrepanReloaded
from core.trepan_original import TrepanOriginalExtractor


@pytest.fixture
def trained_mlp_and_data():
    rng = np.random.default_rng(42)
    n_samples, n_features = 120, 8
    X = rng.random((n_samples, n_features))
    y = (X[:, 0] + X[:, 1] > 1.0).astype(int)
    feature_names = [f"feat_{i}" for i in range(n_features)]
    class_names = ["neg", "pos"]

    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=500, random_state=42)
    mlp.fit(X, y)
    return mlp, X, y, feature_names, class_names


def test_mirror_original_tree_same_predictions(trained_mlp_and_data):
    mlp, X, y, feature_names, class_names = trained_mlp_and_data

    original = TrepanOriginalExtractor()
    original.extract_tree(
        mlp, X, y, sample_size=2000,
        feature_names=feature_names, class_names=class_names,
    )
    original_tree = original.explainer_tree

    trepan = TrepanReloaded(ontology=None, use_default_ontology=False)
    assert not trepan.has_active_ontology
    original_audit = {
        "trepan_fidelity": 0.91,
        "training_target": "MLP Original predictions",
    }
    mirrored_audit = trepan.mirror_original_tree(
        original_tree,
        feature_names=feature_names,
        original_audit=original_audit,
    )

    assert trepan.extractor.explainer_tree is original_tree
    np.testing.assert_array_equal(
        original_tree.predict(X),
        trepan.extractor.explainer_tree.predict(X),
    )
    assert mirrored_audit["mode"] == "mirrored_no_ontology"
    assert mirrored_audit["trepan_fidelity"] == 0.91
    assert mirrored_audit["mirrored_from"] == "Trepan-Original"
    assert trepan.extractor.last_audit == mirrored_audit


def test_no_ontology_metrics_identical_to_original(trained_mlp_and_data):
    mlp, X, y, feature_names, class_names = trained_mlp_and_data
    split = int(len(X) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    original = TrepanOriginalExtractor()
    original.extract_tree(
        mlp, X_train, y_train, sample_size=2000,
        feature_names=feature_names, class_names=class_names,
    )
    original_tree = original.explainer_tree

    trepan = TrepanReloaded(ontology=None, use_default_ontology=False)
    trepan.mirror_original_tree(original_tree, feature_names=feature_names)
    reloaded_tree = trepan.extractor.explainer_tree

    comparator = MetricsComparator()
    results = comparator.compare_all_models(
        mlp_model=mlp,
        trepan_original_tree=original_tree,
        trepan_reloaded_tree=reloaded_tree,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        class_names=class_names,
        ontology_active=False,
    )

    prec_orig = results["precision"]["trepan_original"]
    prec_rel = results["precision"]["trepan_reloaded"]
    fid_orig = results["fidelity"]["trepan_original"]
    fid_rel = results["fidelity"]["trepan_reloaded"]

    assert prec_orig["accuracy"] == prec_rel["accuracy"]
    assert fid_orig["overall_fidelity"] == fid_rel["overall_fidelity"]
    np.testing.assert_array_equal(prec_orig["predictions"], prec_rel["predictions"])
