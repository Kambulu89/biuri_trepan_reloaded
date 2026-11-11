"""Visualização da árvore: cada modelo mostra tamanho total e profundidade máxima, corretos e atualizados ao trocar de modelo."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pytest.importorskip("PyQt6")
from PyQt6.QtWidgets import QApplication
from sklearn.tree import DecisionTreeClassifier

from core.c45_j48_tree import C45Classifier
from core.trepan_original import TrepanOriginalClassifier
from gui.biuri_app_complete import TreeVisualizationWidget
from gui.tree_viz import strings

NAMES = [f"f{i}" for i in range(6)]


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = ((X[:, 0] * X[:, 1] + X[:, 2] + 0.4 * rng.normal(size=300)) > 0).astype(int)
    return X, y


class _Oracle:
    def predict(self, X):
        X = np.asarray(X)
        return ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).astype(int)


def _trepan(data, max_nodes):
    X, _ = data
    return TrepanOriginalClassifier(min_sample=500, max_queries=60000, max_nodes=max_nodes, random_state=7).fit(
        X, oracle=_Oracle(), feature_names=NAMES)


def _expected(tree):
    """Contagens feitas diretamente na estrutura da árvore (independentes da GUI)."""
    if getattr(tree, "root_", None) is None:
        t = tree.tree_
        return int(t.node_count), int(t.n_leaves), int(t.max_depth)
    nodes, leaves, depth, stack = 0, 0, 0, [(tree.root_, 0)]
    while stack:
        n, d = stack.pop()
        nodes += 1
        depth = max(depth, d)
        kids = list(n.children.values()) if isinstance(getattr(n, "children", None), dict) else [
            c for c in (getattr(n, "false_child", None), getattr(n, "true_child", None)) if c is not None]
        if not kids:
            leaves += 1
        stack.extend((c, d + 1) for c in kids)
    return nodes, leaves, depth


def test_info_is_correct_for_sklearn_like_trepan_and_c45_trees(qapp, data):
    X, y = data
    sk_small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    sk_big = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    trepan = _trepan(data, 15)
    c45 = C45Classifier(min_samples_split=2, min_samples_leaf=1).fit(X, (_Oracle().predict(X)))
    widget = TreeVisualizationWidget(
        tree_model=sk_small, feature_names=NAMES, class_names=["0", "1"],
        trepan_original_tree=trepan, trepan_reloaded_tree=sk_big, c45_tree=c45)
    assert widget.tree_options, "o seletor de modelos devia existir"
    seen = set()
    for idx, (name, tree) in enumerate(widget.tree_options):
        widget.tree_combo.setCurrentIndex(idx)
        nodes, leaves, depth = _expected(tree)
        st = widget.tree_widget.tree_stats()
        assert (st["nodes"], st["leaves"], st["depth"]) == (nodes, leaves, depth), name
        assert st["internal"] == nodes - leaves
        text = widget.tree_info_label.text()
        assert text.startswith(name) and f"{nodes} nodos" in text and f"Profundidad máxima: {depth}" in text
        seen.add((nodes, depth))
    assert len(seen) >= 3                                          # modelos diferentes mostram números diferentes
    widget.close()


def test_single_model_shows_info_with_its_name(qapp, data):
    X, y = data
    sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    widget = TreeVisualizationWidget(c45_tree=sk, feature_names=NAMES, class_names=["0", "1"])
    nodes, leaves, depth = _expected(sk)
    text = widget.tree_info_label.text()
    assert text.startswith("C4.5") and f"{nodes} nodos" in text and f"Profundidad máxima: {depth}" in text
    widget.close()


def test_display_filters_never_change_the_reported_size(qapp, data):
    X, y = data
    sk = DecisionTreeClassifier(max_depth=5, random_state=0).fit(X, y)
    widget = TreeVisualizationWidget(c45_tree=sk, feature_names=NAMES, class_names=["0", "1"])
    before = widget.tree_widget.tree_stats()
    widget.tree_widget.set_complexity_filter(min_samples=50, max_depth=1)     # esconde nós só visualmente
    assert widget.tree_widget.layout.hidden_count > 0
    assert widget.tree_widget.tree_stats() == before
    widget.close()


def test_portuguese_catalog_has_the_same_info_key():
    for lang in ("es", "pt"):
        strings.set_language(lang)
        assert "{" not in strings.tr("tree_info", nodes=7, leaves=4, internal=3, depth=3)
    strings.set_language("es")
    assert strings.tr("tree_info", nodes=7, leaves=4, internal=3, depth=3).startswith("Tamaño total: 7")
