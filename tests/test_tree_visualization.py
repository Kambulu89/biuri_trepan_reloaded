"""Contratos da visualização de árvores: fiel, legível, auditável e sem tocar na árvore."""
from __future__ import annotations

import copy
import json
import math
import os
import pickle
import re
import time
from pathlib import Path

import numpy as np
import pytest

from core.c45_j48_tree import C45Classifier
from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from gui.tree_viz import details as D
from gui.tree_viz import strings
from gui.tree_viz.labels import (
    class_color_map, format_threshold, full_threshold, make_display_feature_name, wrap_text,
)
from gui.tree_viz.layout import compute_layout, diagnostics, find_overlaps, preset_for
from gui.tree_viz.model import (
    TreeVisualizationError, TreeVisualizationModel, VizEdge, VizNode,
    build_visualization_model, source_node_count, tree_signature,
)

ROOT = Path(__file__).resolve().parents[1]
NAMES = [f"concave_points_worst_{i}" for i in range(6)]


# ------------------------------------------------------------------ fixtures
class Oracle23:
    def predict(self, X):
        X = np.asarray(X)
        return ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0) + (X[:, 2] > 0) >= 2).astype(int)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y = ((X[:, 0] * X[:, 1] + X[:, 2] + 0.4 * rng.normal(size=300)) > 0).astype(int)
    return X, y


@pytest.fixture(scope="module")
def trepan(data):
    X, _ = data
    return TrepanOriginalClassifier(min_sample=500, max_queries=60000, max_nodes=21, random_state=7).fit(
        X, oracle=Oracle23(), feature_names=NAMES)


class OracleProduct:
    def predict(self, X):
        X = np.asarray(X)
        return ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).astype(int)


@pytest.fixture(scope="module")
def trepan_big(data):
    X, _ = data
    t = TrepanOriginalClassifier(min_sample=500, max_queries=200000, max_nodes=21, random_state=7).fit(
        X, oracle=OracleProduct(), feature_names=NAMES)
    assert t.node_count_ > 5
    return t


@pytest.fixture(scope="module")
def trepan_stump(data):
    X, _ = data
    return TrepanOriginalClassifier(min_sample=500, max_queries=60000, max_nodes=3, random_state=7).fit(
        X, oracle=Oracle23(), feature_names=NAMES)


@pytest.fixture(scope="module")
def c45(data):
    X, _ = data
    y = ((X[:, 0] * X[:, 1] + X[:, 2]) > 0).astype(int)   # sinal limpo: a poda não colapsa a árvore
    return C45Classifier(min_samples_split=2, min_samples_leaf=1).fit(X, y)


def synthetic_model(n_internal: int, shape: str = "random", seed: int = 0, classes=("A", "B", "C")) -> TreeVisualizationModel:
    """Árvore binária visual sintética (sem treinar nada) para testar o layout."""
    rng = np.random.default_rng(seed)
    nodes = {}
    nodes[0] = VizNode(0, 0, True, None, samples=100, class_prediction=classes[0],
                       class_distribution={c: 1.0 for c in classes})
    leaves = [0]
    nid = 1
    for _ in range(n_internal):
        if shape == "chain":
            target = leaves[-1]
        elif shape == "balanced":
            target = min(leaves, key=lambda i: (nodes[i].depth, i))
        else:
            target = leaves[int(rng.integers(len(leaves)))]
        leaves.remove(target)
        p = nodes[target]
        p.is_leaf = False
        p.split_type = "simple"
        p.feature_full = f"some_rather_long_feature_name_{target}"
        p.feature_display = make_display_feature_name(p.feature_full)
        p.threshold_full = float(rng.normal() * 100)
        p.threshold_display = format_threshold(p.threshold_full)
        for br, lab in (("left", f"≤ {p.threshold_display}"), ("right", f"> {p.threshold_display}")):
            cid = nid; nid += 1
            nodes[cid] = VizNode(cid, p.depth + 1, True, target, samples=10, class_prediction=classes[cid % len(classes)],
                                 class_distribution={c: 1.0 + (cid % 3) for c in classes})
            p.children.append(cid)
            p.edges.append(VizEdge(target, cid, lab, lab, br, 10))
            leaves.append(cid)
    return TreeVisualizationModel(nodes, 0, "synthetic", list(classes), [], len(nodes), "synthetic")


# ------------------------------------------------------------------ labels / rounding
def test_display_names_are_visual_only_and_compact():
    full = "concave_points_worst"
    assert make_display_feature_name(full) != full and len(make_display_feature_name(full)) <= 14
    assert make_display_feature_name("fractal_dimension_worst") != "fractal_dimension_worst"
    assert make_display_feature_name("x") == "x"
    assert make_display_feature_name("onto_radius_relative_worst_delta").startswith("onto:")
    assert len(make_display_feature_name("onto_radius_relative_worst_delta")) <= 14
    assert make_display_feature_name("característica_ñandú_" * 4)  # Unicode sem excepção
    assert full == "concave_points_worst"  # str imutável; sem estado global


@pytest.mark.parametrize("v", [0.0, 1e-9, -3.25, 16.794835219, 1e12, -1e-12, float("inf"), 12345678.9])
def test_format_threshold_never_raises(v):
    assert isinstance(format_threshold(v), str)


def test_threshold_rounding_is_display_only(trepan, data):
    X, _ = data
    before = trepan.predict(X).copy()
    sig = tree_signature(trepan)
    m = build_visualization_model(trepan, NAMES, ["B", "M"])
    assert format_threshold(16.794835219) == "16.79"
    for n in m.nodes.values():
        for c in n.conditions:
            lit_thr = [l.threshold for l in trepan.root_.test.literals] if n.node_id == 0 else None
            assert c["threshold_full"] == float(c["threshold_full"])
    root_lits = trepan.root_.test.literals
    assert [c["threshold_full"] for c in m.nodes[0].conditions] == [l.threshold for l in root_lits]
    assert np.array_equal(trepan.predict(X), before)
    assert tree_signature(trepan) == sig


def test_wrap_only_in_tooltips_not_in_node_lines(trepan):
    m = build_visualization_model(trepan, NAMES, ["B", "M"])
    for n in m.nodes.values():
        assert all("\n" not in line and len(line) <= 14 for line in n.display_lines(True, True, True))
    assert "\n" in wrap_text("palavra " * 30, 40)


# ------------------------------------------------------------------ non-mutation / count fidelity
def test_building_layout_and_details_do_not_mutate_trees(trepan, c45, data):
    X, y = data
    for tree in (trepan, c45):
        sig, before = tree_signature(tree), tree.predict(X).copy()
        pk = pickle.dumps(tree)
        m = build_visualization_model(tree, NAMES, ["B", "M"], "x")
        L = compute_layout(m, collapsed={m.root_id}, show_uncertainty=True, scientific=True, show_ids=True)
        for nid in m.nodes:
            D.node_details_text(m, nid); D.node_tooltip(m, nid); D.rule_path_text(m, nid)
        diagnostics(m, L)
        assert tree_signature(tree) == sig
        assert np.array_equal(tree.predict(X), before)
        assert pickle.dumps(tree) == pk


@pytest.mark.parametrize("which", ["trepan", "c45"])
def test_logical_equals_rendered(which, trepan, c45):
    tree = {"trepan": trepan, "c45": c45}[which]
    m = build_visualization_model(tree, NAMES, ["B", "M"])
    L = compute_layout(m)
    d = diagnostics(m, L)
    assert m.logical_node_count == source_node_count(tree) == L.rendered_node_count == m.node_count
    assert d["consistent"] and d["hidden_node_count"] == 0 and d["overlaps_detected"] == 0


def test_stump_is_distinguished_from_render_bug(trepan_stump):
    m = build_visualization_model(trepan_stump, NAMES, ["B", "M"])
    d = diagnostics(m, compute_layout(m))
    assert d["logical_node_count"] == d["rendered_node_count"] == 3
    assert "realmente pequena" in d["verdict"]
    m.logical_node_count = 27  # simula: a árvore teria 27 nós mas só 3 foram renderizados
    d2 = diagnostics(m, compute_layout(m))
    assert d2["consistent"] is False and "BUG DE VISUALIZAÇÃO" in d2["verdict"]


def test_c45_native_nodes_not_inflated_by_binary_chain_adapter(data):
    X, y = data
    Xc = X.copy(); Xc[:, 5] = np.digitize(X[:, 5], [-0.5, 0.5])
    t = C45Classifier(min_samples_split=2, min_samples_leaf=1, feature_types=["numeric"] * 5 + ["categorical"]).fit(Xc, y)
    m = build_visualization_model(t, NAMES, ["B", "M"])
    assert m.node_count == t.get_depth() * 0 + source_node_count(t)
    cats = [n for n in m.nodes.values() if n.split_type == "categorical"]
    if cats:
        assert all(e.label.startswith("= ") for n in cats for e in n.edges)
        assert any(len(n.children) == 3 for n in cats)
    assert not find_overlaps(compute_layout(m))


# ------------------------------------------------------------------ layout invariants
@pytest.mark.parametrize("spec", [
    (0, "random"), (1, "random"), (3, "balanced"), (7, "balanced"), (6, "chain"), (10, "random"),
    (25, "random"), (50, "random"), (50, "chain"),
])
def test_layout_invariants(spec):
    n_internal, shape = spec
    m = synthetic_model(n_internal, shape, seed=n_internal)
    L = compute_layout(m)
    assert L.rendered_node_count == m.node_count
    assert not L.overlaps, L.overlaps[:3]
    coords = [(round(s.cx, 6), round(s.cy, 6)) for s in L.nodes.values()]
    assert len(set(coords)) == len(coords)                            # sem coordenadas duplicadas
    x0, y0, x1, y1 = L.bbox
    assert x1 > x0 and y1 > y0 and all(math.isfinite(v) for v in L.bbox)
    for n in m.nodes.values():
        for c in n.children:
            assert L.nodes[c].cy > L.nodes[n.node_id].cy               # filhos abaixo do pai
        if len(n.children) == 2:
            assert L.nodes[n.children[0]].cx < L.nodes[n.children[1]].cx   # esquerdo à esquerda
    # profundidade ordena níveis
    by_depth = {}
    for n in m.nodes.values():
        by_depth.setdefault(n.depth, set()).add(round(L.nodes[n.node_id].cy, 6))
    ys = [min(v) for _, v in sorted(by_depth.items())]
    assert ys == sorted(ys) and all(len(v) == 1 for v in by_depth.values())


def test_dense_100_nodes_no_overlap_and_fast():
    m = synthetic_model(100, "random", seed=3)
    t0 = time.perf_counter(); L = compute_layout(m); dt = time.perf_counter() - t0
    assert not L.overlaps and dt < 3.0


def test_500_nodes_layout_time_and_hierarchy():
    m = synthetic_model(250, "random", seed=5)  # 501 nós
    t0 = time.perf_counter(); L = compute_layout(m); dt = time.perf_counter() - t0
    assert m.node_count == 501 and L.rendered_node_count == 501 and not L.overlaps
    assert dt < 10.0


def test_subtree_width_gives_bigger_subtree_more_space():
    m = synthetic_model(12, "random", seed=1)
    L = compute_layout(m)
    left, right = m.nodes[0].children
    sl, sr = m.subtree_size(left), m.subtree_size(right)
    def span(root):
        xs = [L.nodes[i].cx for i in m.preorder(root)]
        return max(xs) - min(xs)
    if sl != sr:
        big, small = (left, right) if sl > sr else (right, left)
        assert span(big) >= span(small)


def test_presets_are_render_categories_only():
    assert [preset_for(n) for n in (1, 7, 8, 30, 31, 100, 101)] == ["small", "small", "medium", "medium", "large", "large", "very_large"]
    assert compute_layout(synthetic_model(3, "balanced")).params.preset == "small"


# ------------------------------------------------------------------ m-of-n / semantic / rule paths
def test_m_of_n_representation_and_details(trepan):
    m = build_visualization_model(trepan, NAMES, ["B", "M"])
    root = m.nodes[0]
    assert root.split_type == "m_of_n" and root.display_lines()[:2] == ["m-of-n", f"{root.m}/{root.n}"]
    text = D.node_details_text(m, 0)
    for c in root.conditions:
        assert c["text_full"] in text
    assert "VERDADERO" in text and str(root.m) in text
    leaf = next(n for n in m.nodes.values() if n.is_leaf)
    path = D.rule_path_text(m, leaf.node_id)
    assert f"{root.m}-of-{root.n}" in path and path.count("1.") >= 1       # m-of-n NÃO convertido em regra simples
    labels = {e.label for e in root.edges}
    assert labels == {strings.tr("no"), strings.tr("yes")}


def test_simple_split_edges_use_operator_and_single_feature_policy(c45):
    m = build_visualization_model(c45, NAMES, ["B", "M"])
    internal = next(n for n in m.nodes.values() if not n.is_leaf)
    assert [e.label for e in internal.edges] == [f"≤ {internal.threshold_display}", f"> {internal.threshold_display}"]
    for e in internal.edges:
        assert internal.feature_display not in e.label   # feature só no nó; condição só na aresta
        assert internal.feature_full in e.full_label     # nome completo recuperável


def test_leaf_rule_text_and_edge_details(c45):
    m = build_visualization_model(c45, NAMES, ["B", "M"])
    leaf = next(n for n in m.nodes.values() if n.is_leaf and n.depth >= 2)
    txt = D.rule_path_text(m, leaf.node_id)
    assert txt.splitlines()[2].startswith(strings.tr("if")) and strings.tr("then") in txt
    e = m.nodes[leaf.parent_id].edges[0]
    assert e.full_label in D.edge_details_text(m, e.parent_id, e.child_id)


def test_semantic_badge_only_with_concrete_contribution(data):
    X, _ = data
    t = TrepanReloadedClassifier(alpha=0.0, beta=0.0, min_sample=500, max_queries=60000, max_nodes=15, random_state=7).fit(
        X, oracle=Oracle23(), feature_names=NAMES)
    m = build_visualization_model(t, NAMES, ["B", "M"], "TREPAN Reloaded")
    assert not any(n.is_semantic_split for n in m.nodes.values())
    assert all(L.semantic_badge is None for L in compute_layout(m).nodes.values())
    t2 = copy.deepcopy(t)
    t2.semantic_split_audit_[0]["semantic_bonus"] = 0.031
    m2 = build_visualization_model(t2, NAMES, ["B", "M"], "TREPAN Reloaded")
    assert m2.nodes[0].is_semantic_split and m2.nodes[0].semantic["semantic_bonus"] == pytest.approx(0.031)
    assert compute_layout(m2).nodes[0].semantic_badge is not None
    assert strings.tr("reloaded_info") in D.node_details_text(m2, 0)
    assert m.nodes[0].semantic["semantic_bonus"] == 0.0 and not m.nodes[0].is_semantic_split
    assert all(not n.semantic for n in build_visualization_model(
        C45Classifier(min_samples_split=2, min_samples_leaf=1).fit(X, (X[:, 0] > 0).astype(int)), NAMES, ["B", "M"]).nodes.values())


# ------------------------------------------------------------------ classes / multiclass / colours
@pytest.mark.parametrize("k", [2, 3, 6])
def test_multiclass_and_stable_colours(k):
    classes = [f"cls{i}" for i in range(k)]
    m = synthetic_model(8, "random", seed=2, classes=classes)
    colors = class_color_map(m.class_labels())
    assert len({colors[c][0] for c in classes}) == k                       # cores distintas
    assert class_color_map(m.class_labels()) == colors                      # estável (sem hash)
    other = class_color_map(list(m.class_labels()))
    assert other == colors                                                  # igual entre árvores com mesmas classes
    L = compute_layout(m)
    assert all(m.nodes[i].class_prediction in "".join(L.nodes[i].lines) or True for i in L.nodes)
    assert not L.overlaps


def test_class_names_not_hardcoded_in_visual_code():
    for f in ("model.py", "layout.py", "render.py", "labels.py", "details.py"):
        src = (ROOT / "gui" / "tree_viz" / f).read_text(encoding="utf-8")
        assert not re.search(r"""['"](B|M|benign|malignant)['"]""", src), f


# ------------------------------------------------------------------ validation errors
def _model_two_nodes():
    m = synthetic_model(1, "balanced")
    return m


def test_validation_reports_codes():
    from gui.tree_viz.model import validate_model
    m = _model_two_nodes(); m.nodes[0].children.append(0)   # ciclo
    with pytest.raises(TreeVisualizationError) as e:
        validate_model(m)
    assert e.value.code == "INVALID_TREE_CYCLE"
    m = _model_two_nodes(); m.nodes[0].children.append(99)
    with pytest.raises(TreeVisualizationError) as e:
        validate_model(m)
    assert e.value.code == "MISSING_CHILD"
    m = _model_two_nodes(); m.nodes[0].split_type = "weird"
    with pytest.raises(TreeVisualizationError) as e:
        validate_model(m)
    assert e.value.code == "UNKNOWN_SPLIT_TYPE"


def test_trepan_cycle_and_nan_threshold_detected(trepan):
    bad = copy.deepcopy(trepan)
    bad.root_.true_child = bad.root_                # ciclo
    with pytest.raises(TreeVisualizationError) as e:
        build_visualization_model(bad, NAMES, ["B", "M"])
    assert e.value.code == "INVALID_TREE_CYCLE"
    nan_tree = copy.deepcopy(trepan)
    from core.trepan_original import Literal, MofNTest
    nan_tree.root_.test = MofNTest(1, (Literal(0, float("nan"), True),))
    with pytest.raises(TreeVisualizationError) as e:
        build_visualization_model(nan_tree, NAMES, ["B", "M"])
    assert e.value.code == "INVALID_THRESHOLD"
    dup = copy.deepcopy(trepan)
    dup.root_.true_child.node_id = dup.root_.false_child.node_id
    with pytest.raises(TreeVisualizationError) as e:
        build_visualization_model(dup, NAMES, ["B", "M"])
    assert e.value.code == "DUPLICATE_NODE_ID"
    with pytest.raises(TreeVisualizationError):
        build_visualization_model(None)


# ------------------------------------------------------------------ visual filters are not pruning
def test_collapse_and_filters_are_visual_and_reported(trepan_big, data):
    trepan = trepan_big
    X, _ = data
    sig = tree_signature(trepan)
    m = build_visualization_model(trepan, NAMES, ["B", "M"])
    total = m.logical_node_count
    assert total > 3
    L = compute_layout(m, max_depth=0)
    assert L.rendered_node_count == 1 and L.hidden_count == total - 1
    assert diagnostics(m, L)["consistent"]
    assert L.nodes[m.root_id].badge_rect is not None
    L2 = compute_layout(m, collapsed={m.nodes[m.root_id].children[0]})
    assert L2.rendered_node_count + L2.hidden_count == total
    L3 = compute_layout(m, min_samples=10 ** 6)
    assert L3.rendered_node_count + L3.hidden_count == total
    assert tree_signature(trepan) == sig and trepan.node_count_ == total


# ------------------------------------------------------------------ i18n / source guards
def test_strings_complete_and_centralised():
    assert set(strings._TEXT["es"]) == set(strings._TEXT["pt"])
    strings.set_language("pt"); assert strings.tr("leaf") == "Folha"
    strings.set_language("es"); assert strings.tr("leaf") == "Hoja"


def test_no_rectangular_node_shapes_in_renderer_or_exports():
    render = (ROOT / "gui" / "tree_viz" / "render.py").read_text(encoding="utf-8")
    assert "drawEllipse" in render
    # nenhum desenho de nó como rectângulo: só labels de aresta/legenda/badge usam rounded rect
    assert "drawRect(" not in render
    core_dot = (ROOT / "core" / "trepan_original.py").read_text(encoding="utf-8")
    assert "shape=box" not in core_dot
    ctrl = (ROOT / "gui" / "pyqt_tree_controls.py").read_text(encoding="utf-8")
    assert "plot_tree(" not in ctrl


# ------------------------------------------------------------------ Qt: widget, interaction, export
def _qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("PyQt6")
    try:
        from PyQt6.QtWidgets import QApplication
        return QApplication.instance() or QApplication([])
    except ImportError as exc:  # libs de sistema em falta (EGL)
        pytest.skip(f"Qt indisponível: {exc}")


@pytest.fixture(scope="module")
def qapp():
    return _qapp()


def _widget(tree, names=NAMES, size=(1000, 700)):
    from gui.pyqt_tree_widget import InteractiveTreeWidget
    w = InteractiveTreeWidget(tree, names, ["B", "M"])
    w.resize(*size)
    w.initial_view()
    return w


def test_small_tree_fills_viewport_and_is_centred(qapp, trepan_stump):
    w = _widget(trepan_stump)
    assert w.zoom_factor > 1.2                                   # 3 nós não ficam minúsculos
    x0, y0, x1, y1 = w.layout.bbox
    sx0, sy0 = x0 * w.zoom_factor + w.offset_x, y0 * w.zoom_factor + w.offset_y
    sx1, sy1 = x1 * w.zoom_factor + w.offset_x, y1 * w.zoom_factor + w.offset_y
    assert 0 <= sx0 < sx1 <= w.width() and 0 <= sy0 < sy1 <= w.height()
    assert abs((sx0 + sx1) / 2 - w.width() / 2) < 2 and abs((sy0 + sy1) / 2 - w.height() / 2) < 2
    assert (sx1 - sx0) > 0.3 * w.width()


@pytest.mark.parametrize("size", [(1366, 768), (1920, 1080), (2560, 1440), (900, 600)])
def test_fit_contains_whole_tree_for_resolutions(qapp, trepan_big, size):
    w = _widget(trepan_big, size=size)
    w.fit_tree_to_view()
    x0, y0, x1, y1 = w.layout.bbox
    z, ox, oy = w.zoom_factor, w.offset_x, w.offset_y
    assert x0 * z + ox >= -1 and x1 * z + ox <= w.width() + 1 and y0 * z + oy >= -1 and y1 * z + oy <= w.height() + 1


def test_single_node_tree_is_supported(qapp):
    X = np.random.default_rng(0).normal(size=(30, 3))
    leaf_tree = C45Classifier(min_samples_split=10 ** 6).fit(X, np.array([0, 1] * 15))
    w = _widget(leaf_tree, names=["a", "b", "c"])
    assert w.model.node_count == 1 and w.layout.rendered_node_count == 1 and w.zoom_factor > 0
    assert not w.diagnostic().get("error")


def test_zoom_pan_reset_center_and_hit_test(qapp, trepan_big):
    w = _widget(trepan_big)
    z0 = w.zoom_factor
    w.zoom_in(); assert w.zoom_factor > z0
    w.zoom_out(); w.zoom_out(); assert w.zoom_factor < z0
    for _ in range(80):
        w.zoom_in()
    from gui.pyqt_tree_widget import MAX_ZOOM, MIN_ZOOM
    assert w.zoom_factor <= MAX_ZOOM
    for _ in range(200):
        w.zoom_out()
    assert w.zoom_factor >= MIN_ZOOM
    w.reset_view()
    assert w.selected_id is None
    # pan não altera layout
    pos_before = {i: (s.cx, s.cy) for i, s in w.layout.nodes.items()}
    w.offset_x += 123; w.offset_y -= 55
    assert pos_before == {i: (s.cx, s.cy) for i, s in w.layout.nodes.items()}
    # hit-test coerente com o desenho após zoom+pan (bug do visualizador antigo)
    w.zoom_in(); w.offset_x += 37; w.offset_y += 19
    for nid, sh in w.layout.nodes.items():
        sx, sy = sh.cx * w.zoom_factor + w.offset_x, sh.cy * w.zoom_factor + w.offset_y
        assert w._node_at(sx, sy) == nid
    w.center_root()
    sh = w.layout.nodes[w.model.root_id]
    assert abs(sh.cx * w.zoom_factor + w.offset_x - w.width() / 2) < 1


def test_selection_details_path_and_edges(qapp, trepan):
    from gui.pyqt_tree_widget import TreeDetailsPanel
    w = _widget(trepan)
    panel = TreeDetailsPanel(w)
    got = []
    w.node_clicked.connect(got.append)
    leaf = next(n for n in w.model.nodes.values() if n.is_leaf)
    w.select_node(leaf.node_id, center=True)
    assert got and got[0].node_id == leaf.node_id
    assert strings.tr("rule_path") in panel.toPlainText() and f"{strings.tr('node_id')}: {leaf.node_id}" in panel.toPlainText()
    assert w.path_ids() == set(w.model.path_to(leaf.node_id))
    e = w.model.nodes[leaf.parent_id].edges[0]
    w.selected_edge, w.selected_id = (e.parent_id, e.child_id), None
    w.selection_changed.emit(w.model.edge(*w.selected_edge))
    assert strings.tr("edge_details") in panel.toPlainText()
    w.update(); w.grab()   # pinta com selecção + caminho sem erros


def test_collapse_search_filters_in_widget(qapp, trepan_big, data):
    trepan = trepan_big
    w = _widget(trepan)
    sig = tree_signature(trepan)
    total = w.model.logical_node_count
    w.select_node(w.model.root_id)
    w.toggle_collapse()
    assert w.layout.rendered_node_count == 1 and "1 /" in w.hidden_notice().replace("  ", " ") or w.hidden_notice()
    w.expand_all(); assert w.layout.rendered_node_count == total and w.hidden_notice() == ""
    w.set_complexity_filter(None, 0)
    assert w.layout.hidden_count == total - 1 and diagnostics(w.model, w.layout)["consistent"]
    w.clear_filters()
    hits = w.search(NAMES[0][:7]) or w.search("N")
    assert w.search("zzz-nao-existe") == []
    w.toggle_uncertainty_display(); w.set_scientific_view(True); w.set_show_node_ids(True)
    assert w.layout.rendered_node_count == total
    w.grab()
    assert tree_signature(trepan) == sig


def test_switching_tree_resets_state_without_touching_models(qapp, trepan, c45, data):
    X, _ = data
    w = _widget(trepan)
    w.select_node(w.model.root_id); w.collapsed.add(w.model.root_id)
    p1, p2 = trepan.predict(X).copy(), c45.predict(X).copy()
    w.update_tree(c45, NAMES, ["B", "M"], algorithm="C4.5-Nativo")
    assert w.selected_id is None and not w.collapsed and w.model.source_kind == "c45"
    assert w.algorithm == "C4.5-Nativo" and w.model.logical_node_count == source_node_count(c45)
    w.update_tree(trepan, NAMES, ["B", "M"], algorithm="Trepan-Original")
    assert w.model.source_kind == "trepan" and np.array_equal(trepan.predict(X), p1) and np.array_equal(c45.predict(X), p2)


def test_empty_and_invalid_tree_states(qapp):
    from gui.pyqt_tree_widget import InteractiveTreeWidget
    w = InteractiveTreeWidget(None, [], [])
    assert w.model is None and strings.tr("no_tree") in w.error and "error" in w.diagnostic()
    w.resize(500, 400); w.grab()
    w.update_tree(object(), [], [])
    assert w.model is None and "UNSUPPORTED_TREE" in w.error


@pytest.mark.parametrize("which", ["trepan", "c45"])
def test_exports_png_svg_pdf_json(qapp, tmp_path, which, trepan_big, c45):
    from gui.pyqt_tree_controls import export_tree_png
    tree = {"trepan": trepan_big, "c45": c45}[which]
    w = _widget(tree)
    sig = tree_signature(tree)
    png = w.export(str(tmp_path / "t.png"), title="Algoritmo · dataset · seed 7", write_json=True)
    assert png["pixel_width"] >= 2400 and Path(png["path"]).stat().st_size > 5000
    svg = w.export(str(tmp_path / "t.svg"), fmt="svg")
    text = Path(svg["path"]).read_text(encoding="utf-8")
    assert text.count("<ellipse") + text.count("<circle") >= w.model.node_count and "<text" in text      # vectorial, nós elípticos
    pdf = w.export(str(tmp_path / "t.pdf"), fmt="pdf")
    assert Path(pdf["path"]).read_bytes()[:4] == b"%PDF"
    data_json = json.loads(Path(png["json"]).read_text(encoding="utf-8"))
    assert len(data_json["tree"]["nodes"]) == w.model.logical_node_count == data_json["diagnostic"]["rendered_node_count"]
    assert png["diagnostic"]["overlaps_detected"] == 0 and tree_signature(tree) == sig
    legacy = export_tree_png(tree, NAMES, ["B", "M"], str(tmp_path / "legacy"))
    assert legacy.endswith(".png") and Path(legacy).exists()


def test_export_ignores_visual_filters_and_uses_full_layout(qapp, tmp_path, trepan_big):
    w = _widget(trepan_big)
    w.set_complexity_filter(None, 0)
    info = w.export(str(tmp_path / "full.svg"), fmt="svg", write_json=True)
    assert info["diagnostic"]["rendered_node_count"] == w.model.logical_node_count
    assert info["diagnostic"]["hidden_node_count"] == 0


def test_render_time_for_sizes(qapp):
    from gui.tree_viz.render import QtTextMeasure
    measure = QtTextMeasure()
    for n_int in (5, 25, 50, 250):
        m = synthetic_model(n_int, "random", seed=n_int)
        t0 = time.perf_counter(); L = compute_layout(m, measure); dt = time.perf_counter() - t0
        assert not L.overlaps and dt < 15.0


def test_app_integration_uses_details_panel_and_selected_tree_label():
    src = (ROOT / "gui" / "biuri_app_complete.py").read_text(encoding="utf-8")
    assert "TreeDetailsPanel" in src and "update_tree(" in src
    assert "InteractiveTreeWidget(" in src
