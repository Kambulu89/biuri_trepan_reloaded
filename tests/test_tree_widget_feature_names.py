"""Regressão: visualização da árvore com n_features > len(feature_names ARFF)."""
import sys
import pytest
pytest.importorskip("PyQt6", reason="dependência opcional não instalada")
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.pyqt_tree_widget import InteractiveTreeWidget


def test_align_feature_names_pads_for_reloaded_tree():
    base_names = [f"feat_{i}" for i in range(30)]
    full_names = base_names + [f"onto_{i}" for i in range(182)]
    rng = np.random.default_rng(0)
    X = rng.random((50, 212))
    y = rng.integers(0, 2, size=50)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)

    names = InteractiveTreeWidget._align_feature_names(full_names, tree)
    assert len(names) == 212
    assert names[30] == "onto_0"

    names_short = InteractiveTreeWidget._align_feature_names(base_names, tree)
    assert len(names_short) == 212
    assert names_short[31] == "feature_31"


def test_class_names_for_tree_subset():
    tree = DecisionTreeClassifier(max_depth=1, random_state=0)
    tree.fit([[0], [1], [2], [3]], [0, 0, 0, 0])
    export = InteractiveTreeWidget._class_names_for_tree(tree, ["benign", "malignant"])
    assert len(export) == 1
    assert export[0] == "benign"
import pytest
pytest.importorskip("PyQt6", reason="dependência opcional não instalada")
