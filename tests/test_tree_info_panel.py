"""Visualização da árvore: cada modelo mostra tamanho total e profundidade máxima, corretos e atualizados ao trocar de modelo."""
import os
from pathlib import Path

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


# ---- a exportação da imagem de um modelo inclui tamanho total e profundidade máxima
def _png_dark_rows(path, y0, y1):
    from PyQt6.QtGui import QImage
    img = QImage(str(path))
    dark = 0
    for y in range(max(0, y0), min(img.height(), y1)):
        for x in range(0, img.width(), 2):
            c = img.pixelColor(x, y)
            if c.red() < 140 and c.green() < 140 and c.blue() < 140:
                dark += 1
    return dark, img


def test_exported_image_of_each_model_carries_size_and_depth(qapp, data, tmp_path):
    import json
    X, y = data
    small = DecisionTreeClassifier(max_depth=2, random_state=0).fit(X, y)
    big = DecisionTreeClassifier(max_depth=6, random_state=0).fit(X, y)
    widget = TreeVisualizationWidget(trepan_original_tree=small, trepan_reloaded_tree=big, feature_names=NAMES,
                                     class_names=["0", "1"])
    for idx, (name, tree) in enumerate(widget.tree_options):
        widget.tree_combo.setCurrentIndex(idx)
        nodes, leaves, depth = _expected(tree)
        info = widget.tree_widget.export(str(tmp_path / f"m{idx}.png"), fmt="png", title=name, write_json=True)
        assert info["stats"] == {"nodes": nodes, "leaves": leaves, "internal": nodes - leaves, "depth": depth}
        assert f"{nodes} nodos" in info["stats_text"] and f"Profundidad máxima: {depth}" in info["stats_text"]
        saved = json.loads(Path(info["json"]).read_text(encoding="utf-8"))
        assert saved["tree"]["stats"]["nodes"] == nodes and saved["tree"]["stats"]["depth"] == depth
    widget.close()


def test_png_has_an_extra_header_line_with_the_info(qapp, data, tmp_path):
    from gui.pyqt_tree_widget import InteractiveTreeWidget
    X, y = data
    sk = DecisionTreeClassifier(max_depth=4, random_state=0).fit(X, y)
    w = InteractiveTreeWidget(sk, NAMES, ["0", "1"], algorithm="Modelo")
    on = w.export(str(tmp_path / "on.png"), fmt="png", title=None)
    from gui.tree_viz.render import export_tree
    off = export_tree(w.model, str(tmp_path / "off.png"), fmt="png", title=None, show_stats=False)
    # faixa do cabeçalho (mundo 8..26): acima da árvore, que só começa na margem (28)
    dark_on, img_on = _png_dark_rows(tmp_path / "on.png", int(8 * on["scale"]), int(26 * on["scale"]))
    dark_off, img_off = _png_dark_rows(tmp_path / "off.png", int(8 * off["scale"]), int(26 * off["scale"]))
    assert img_on.height() > img_off.height()                    # a linha de informação ocupa espaço próprio
    assert dark_on > 0 and dark_off == 0                         # e tem texto desenhado onde antes estava a margem em branco
    w.close()


def test_svg_export_contains_the_info_text(qapp, data, tmp_path):
    X, y = data
    sk = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    widget = TreeVisualizationWidget(c45_tree=sk, feature_names=NAMES, class_names=["0", "1"])
    info = widget.tree_widget.export(str(tmp_path / "t.svg"), fmt="svg", title="C4.5")
    nodes, leaves, depth = _expected(sk)
    svg = Path(info["path"]).read_text(encoding="utf-8")
    assert "Profundidad" in svg or f"{depth}" in svg                # texto (ou glifos) presente no SVG
    assert info["stats"]["depth"] == depth and info["stats"]["nodes"] == nodes
    widget.close()
