"""Trepan-Reloaded: dominância é informativa e não substitui a árvore extraída."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trepan_reloaded_extractor import TrepanReloadedExtractor


def test_report_dominance_status_does_not_replace_tree():
    rng = np.random.default_rng(0)
    n = 40
    X_base = rng.random((n, 4))
    X_aug = np.hstack([X_base, rng.random((n, 2))])
    y = rng.integers(0, 2, size=n)

    mlp_orig = MLPClassifier(max_iter=200, random_state=0)
    mlp_orig.fit(X_base, y)
    mlp_onto = MLPClassifier(max_iter=200, random_state=1)
    mlp_onto.fit(X_aug, y)

    tree_reloaded = DecisionTreeClassifier(max_depth=3, random_state=5)
    tree_reloaded.fit(X_aug, y)
    tree_orig = DecisionTreeClassifier(max_depth=3, random_state=6)
    tree_orig.fit(X_base, y)
    tree_c45 = DecisionTreeClassifier(max_depth=3, random_state=7)
    tree_c45.fit(X_base, y)

    ext = TrepanReloadedExtractor(ontology=object())
    ext.ontology_active = True
    ext.explainer_tree = tree_reloaded
    ext._training_cache = {'feature_names': [f"f{i}" for i in range(6)]}
    ext._original_feature_names = [f"f{i}" for i in range(4)]
    ext._base_feature_names = [f"f{i}" for i in range(4)]
    ext._mlp_model_onto = mlp_onto

    before_id = id(ext.explainer_tree)
    result = ext.report_ontology_dominance_status(
        mlp_orig, X_aug[:30], y[:30],
        [f"f{i}" for i in range(6)], ["0", "1"],
        tree_orig, tree_c45,
        X_eval=X_aug[30:], y_eval=y[30:],
        mlp_model_onto=mlp_onto,
        dominance_check_enabled=True,
    )

    assert result is tree_reloaded
    assert id(ext.explainer_tree) == before_id
    assert ext._last_dominance_metrics is not None
    assert ext._last_dominance_informational is True
