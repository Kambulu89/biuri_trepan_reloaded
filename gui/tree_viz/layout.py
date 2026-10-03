"""Layout hierárquico (Reingold–Tilford com contornos) para nós ELÍPTICOS de largura variável.

Não depende de Qt. Recebe um ``TreeVisualizationModel`` (só leitura) e devolve geometria:
elipses de nós, segmentos de aresta, caixas de labels, badges e a bounding box real.
Colisões são detectadas após o posicionamento e resolvidas aumentando o espaçamento
(nunca movendo nós aleatoriamente nem apagando nada).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Set, Tuple

from gui.tree_viz.model import TreeVisualizationModel, VizNode
from gui.tree_viz.strings import tr

Rect = Tuple[float, float, float, float]  # x0, y0, x1, y1


class TextMeasure:
    """Medidor de texto injectável (o renderer Qt fornece métricas reais)."""

    def width(self, text: str, px: float, bold: bool = False) -> float:
        return len(text) * px * (0.62 if bold else 0.56)


@dataclass
class LayoutParams:
    preset: str = "medium"
    font_px: float = 14.0
    small_font_px: float = 12.0
    label_font_px: float = 12.0
    h_gap: float = 30.0
    v_gap: float = 62.0
    min_radius: float = 22.0
    initial_zoom_cap: float = 2.4
    min_legible_zoom: float = 0.55


PRESETS = {
    "small": dict(font_px=17.0, small_font_px=14.5, label_font_px=14.0, h_gap=46.0, v_gap=76.0, min_radius=28.0, initial_zoom_cap=2.6, min_legible_zoom=0.6),
    "medium": dict(font_px=14.0, small_font_px=12.0, label_font_px=12.0, h_gap=30.0, v_gap=64.0, min_radius=24.0, initial_zoom_cap=1.8, min_legible_zoom=0.55),
    "large": dict(font_px=13.0, small_font_px=11.0, label_font_px=11.0, h_gap=24.0, v_gap=58.0, min_radius=22.0, initial_zoom_cap=1.2, min_legible_zoom=0.5),
    "very_large": dict(font_px=12.0, small_font_px=10.5, label_font_px=10.5, h_gap=20.0, v_gap=52.0, min_radius=20.0, initial_zoom_cap=1.0, min_legible_zoom=0.45),
}


def preset_for(node_count: int) -> str:
    """Categorias só de rendering: <=7, 8-30, 31-100, >100 nós visíveis."""
    if node_count <= 7:
        return "small"
    if node_count <= 30:
        return "medium"
    if node_count <= 100:
        return "large"
    return "very_large"


@dataclass
class NodeShape:
    node_id: int
    cx: float
    cy: float
    rx: float
    ry: float
    lines: List[str]
    collapsed_hidden: int = 0           # nº de nós escondidos sob este nó
    badge_rect: Optional[Rect] = None   # "+N nós" por baixo do nó
    semantic_badge: Optional[Tuple[float, float, float]] = None  # (cx, cy, r)

    @property
    def aabb(self) -> Rect:
        return (self.cx - self.rx, self.cy - self.ry, self.cx + self.rx, self.cy + self.ry)


@dataclass
class EdgeShape:
    parent_id: int
    child_id: int
    p0: Tuple[float, float]
    p1: Tuple[float, float]
    label: str
    label_rect: Rect


@dataclass
class Layout:
    nodes: Dict[int, NodeShape]
    edges: List[EdgeShape]
    bbox: Rect
    params: LayoutParams
    overlaps: List[tuple] = field(default_factory=list)
    visible_ids: List[int] = field(default_factory=list)
    hidden_count: int = 0
    iterations: int = 0

    @property
    def rendered_node_count(self) -> int:
        return len(self.nodes)


# ---------------------------------------------------------------------------- geometry
def _rect_overlap(a: Rect, b: Rect, tol: float = 0.0) -> bool:
    return a[0] < b[2] - tol and b[0] < a[2] - tol and a[1] < b[3] - tol and b[1] < a[3] - tol


def _ellipse_rect_overlap(n: NodeShape, r: Rect) -> bool:
    px = min(max(n.cx, r[0]), r[2])
    py = min(max(n.cy, r[1]), r[3])
    return ((px - n.cx) / n.rx) ** 2 + ((py - n.cy) / n.ry) ** 2 < 1.0


def _segment_rect_overlap(p0, p1, r: Rect) -> bool:
    x0, y0 = p0
    dx, dy = p1[0] - x0, p1[1] - y0
    t0, t1 = 0.0, 1.0
    for p, q in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
        if abs(p) < 1e-12:
            if q < 0:
                return False
        else:
            t = q / p
            if p < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
            if t0 > t1:
                return False
    return True


def _segment_ellipse_overlap(p0, p1, n: NodeShape) -> bool:
    ax, ay = (p0[0] - n.cx) / n.rx, (p0[1] - n.cy) / n.ry
    bx, by = (p1[0] - n.cx) / n.rx, (p1[1] - n.cy) / n.ry
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    t = 0.0 if L2 == 0 else max(0.0, min(1.0, -(ax * dx + ay * dy) / L2))
    qx, qy = ax + t * dx, ay + t * dy
    return qx * qx + qy * qy < 0.98 ** 2


def _boundary_point(n: NodeShape, tx: float, ty: float) -> Tuple[float, float]:
    dx, dy = tx - n.cx, ty - n.cy
    d = math.sqrt((dx / n.rx) ** 2 + (dy / n.ry) ** 2)
    if d == 0:
        return (n.cx, n.cy)
    return (n.cx + dx / d, n.cy + dy / d)


# ---------------------------------------------------------------------------- core layout
def visible_set(model: TreeVisualizationModel, collapsed: Set[int], max_depth: Optional[int], min_samples: Optional[int]):
    """Nós visíveis e, para cada nó fronteira, quantos descendentes ficam escondidos.

    FILTROS PURAMENTE VISUAIS: nada é podado nem alterado na árvore; os nós escondidos são
    contabilizados e assinalados (``+N nós``)."""
    hidden_under: Dict[int, int] = {}
    visible: List[int] = []
    stack = [model.root_id]
    while stack:
        nid = stack.pop()
        n = model.nodes[nid]
        visible.append(nid)
        boundary = False
        if n.children:
            if nid in collapsed:
                boundary = True
            elif max_depth is not None and n.depth >= max_depth:
                boundary = True
            elif min_samples is not None and nid != model.root_id and n.samples is not None and n.samples < min_samples:
                boundary = True
        if boundary:
            hidden_under[nid] = model.subtree_size(nid) - 1
            continue
        stack.extend(reversed(n.children))
    return visible, hidden_under


def _build(model, measure, visible, hidden_under, params, hscale, vscale,
           show_uncertainty, scientific, show_ids) -> Layout:
    vis_set = set(visible)
    font, small, lab = params.font_px, params.small_font_px, params.label_font_px
    shapes: Dict[int, NodeShape] = {}
    for nid in visible:
        n = model.nodes[nid]
        lines = n.display_lines(show_uncertainty, scientific, show_ids)
        widths = [measure.width(t, font if i == 0 else small, bold=(i == 0)) for i, t in enumerate(lines)]
        heights = [(font if i == 0 else small) * 1.22 for i in range(len(lines))]
        w, h = max(widths) + 6, sum(heights)
        ry = max(params.min_radius, h / 2 * 1.30)
        rx = max(ry, w / 2 * 1.38)
        sh = NodeShape(nid, 0.0, 0.0, rx, ry, lines, collapsed_hidden=hidden_under.get(nid, 0))
        shapes[nid] = sh
    label_w: Dict[int, float] = {}
    for nid in visible:
        for e in model.nodes[nid].edges:
            if e.child_id in vis_set:
                label_w[e.child_id] = measure.width(e.label, lab) + 10
    hgap, vgap = params.h_gap * hscale, params.v_gap * vscale

    def half(nid: int) -> float:
        h = shapes[nid].rx
        if shapes[nid].collapsed_hidden:
            h = max(h, measure.width(tr("collapsed_more", n=shapes[nid].collapsed_hidden), small) / 2 + 4)
        return max(h, label_w.get(nid, 0) / 2)

    order = _preorder(model, vis_set, hidden_under)
    contours: Dict[int, List[Tuple[float, float]]] = {}
    rel: Dict[int, float] = {}
    for nid in reversed(order):
        kids = [c for c in model.nodes[nid].children if c in vis_set] if nid not in hidden_under else []
        hw = half(nid)
        if not kids:
            contours[nid] = [(-hw, hw)]
            continue
        merged = list(contours[kids[0]])
        pos = [0.0]
        for k in kids[1:]:
            c = contours[k]
            shift = max(merged[d][1] + hgap - c[d][0] for d in range(min(len(merged), len(c))))
            pos.append(shift)
            for d in range(len(c)):
                lo, hi = c[d][0] + shift, c[d][1] + shift
                if d < len(merged):
                    merged[d] = (min(merged[d][0], lo), max(merged[d][1], hi))
                else:
                    merged.append((lo, hi))
        centre = (pos[0] + pos[-1]) / 2
        for k, p in zip(kids, pos):
            rel[k] = p - centre
        contours[nid] = [(-hw, hw)] + [(lo - centre, hi - centre) for lo, hi in merged]

    xs = {model.root_id: 0.0}
    for nid in order:
        for c in model.nodes[nid].children:
            if c in rel and nid in xs:
                xs[c] = xs[nid] + rel[c]
    depth_of = {nid: model.nodes[nid].depth for nid in visible}
    levels: Dict[int, float] = {}
    for nid in visible:
        extra = (small * 1.4 + 4) if shapes[nid].collapsed_hidden else 0.0
        levels[depth_of[nid]] = max(levels.get(depth_of[nid], 0.0), shapes[nid].ry + extra)
    ys: Dict[int, float] = {}
    y = 0.0
    prev = None
    for d in sorted(levels):
        if prev is not None:
            y += levels[prev] + vgap + shapes_ry_max(levels, d, shapes, depth_of)
        else:
            y = shapes_ry_max(levels, d, shapes, depth_of)
        ys[d] = y
        prev = d
    for nid in visible:
        shapes[nid].cx, shapes[nid].cy = xs[nid], ys[depth_of[nid]]
        sh = shapes[nid]
        if sh.collapsed_hidden:
            tw = measure.width(tr("collapsed_more", n=sh.collapsed_hidden), small) + 8
            sh.badge_rect = (sh.cx - tw / 2, sh.cy + sh.ry + 2, sh.cx + tw / 2, sh.cy + sh.ry + 2 + small * 1.4)
        if model.nodes[nid].is_semantic_split:
            r = max(7.0, small * 0.55)
            a = math.radians(-45)
            sh.semantic_badge = (sh.cx + sh.rx * math.cos(a) * 0.92 + r * 0.35, sh.cy + sh.ry * math.sin(a) * 0.92 - r * 0.35, r)

    edges: List[EdgeShape] = []
    lab_h = lab * 1.35
    for nid in visible:
        if nid in hidden_under:
            continue
        for e in model.nodes[nid].edges:
            if e.child_id not in vis_set:
                continue
            p, c = shapes[nid], shapes[e.child_id]
            p0, p1 = _boundary_point(p, c.cx, c.cy), _boundary_point(c, p.cx, p.cy)
            mx, my = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            dx, dy = p1[0] - p0[0], p1[1] - p0[1]
            sign = -1.0 if c.cx < p.cx - 1e-6 else 1.0
            lw = label_w[e.child_id]
            slope = abs(dx / dy) if abs(dy) > 1e-9 else 0.0
            cx_l = mx + sign * (lw / 2 + 3 + slope * lab_h / 2)
            edges.append(EdgeShape(nid, e.child_id, p0, p1, e.label,
                                   (cx_l - lw / 2, my - lab_h / 2, cx_l + lw / 2, my + lab_h / 2)))
    x0 = min([s.aabb[0] for s in shapes.values()] + [e.label_rect[0] for e in edges])
    y0 = min([s.aabb[1] for s in shapes.values()] + [e.label_rect[1] for e in edges])
    x1 = max([s.aabb[2] for s in shapes.values()] + [e.label_rect[2] for e in edges] + [s.badge_rect[2] for s in shapes.values() if s.badge_rect])
    y1 = max([s.aabb[3] for s in shapes.values()] + [e.label_rect[3] for e in edges] + [s.badge_rect[3] for s in shapes.values() if s.badge_rect])
    return Layout(shapes, edges, (x0, y0, x1, y1), params, visible_ids=list(visible),
                  hidden_count=sum(hidden_under.values()))


def shapes_ry_max(levels, d, shapes, depth_of) -> float:
    return levels[d]


def _preorder(model, vis_set, hidden_under) -> List[int]:
    out, stack = [], [model.root_id]
    while stack:
        nid = stack.pop()
        out.append(nid)
        if nid in hidden_under:
            continue
        stack.extend(reversed([c for c in model.nodes[nid].children if c in vis_set]))
    return out


def find_overlaps(layout: Layout) -> List[tuple]:
    """Colisões: node-node, node-label, label-label, edge-label, edge-node (aresta atravessa nó)."""
    out: List[tuple] = []
    nodes = list(layout.nodes.values())
    labels = layout.edges
    cell = max(40.0, 2.2 * max([n.rx for n in nodes] + [1.0]))

    def cells(r: Rect):
        for ix in range(int(r[0] // cell), int(r[2] // cell) + 1):
            for iy in range(int(r[1] // cell), int(r[3] // cell) + 1):
                yield ix, iy

    grid: Dict[Tuple[int, int], List[tuple]] = {}
    for n in nodes:
        for c in cells(n.aabb):
            grid.setdefault(c, []).append(("n", n))
        if n.badge_rect:
            for c in cells(n.badge_rect):
                grid.setdefault(c, []).append(("b", n))
    for i, e in enumerate(labels):
        for c in cells(e.label_rect):
            grid.setdefault(c, []).append(("l", e))
    seen: Set[tuple] = set()
    for bucket in grid.values():
        for ai in range(len(bucket)):
            for bi in range(ai + 1, len(bucket)):
                (ka, a), (kb, b) = bucket[ai], bucket[bi]
                key = (ka, id(a), kb, id(b)) if id(a) < id(b) else (kb, id(b), ka, id(a))
                if key in seen:
                    continue
                seen.add(key)
                if ka == "n" and kb == "n":
                    if _rect_overlap(a.aabb, b.aabb):
                        out.append(("node-node", a.node_id, b.node_id))
                elif ka == "l" and kb == "l":
                    if _rect_overlap(a.label_rect, b.label_rect):
                        out.append(("label-label", a.child_id, b.child_id))
                else:
                    node, lab, kind = (a, b, ka) if kb == "l" else (b, a, kb) if ka == "l" else (None, None, None)
                    if node is not None and kind in ("n", "b"):
                        rect_hit = _ellipse_rect_overlap(node, lab.label_rect) if kind == "n" else _rect_overlap(node.badge_rect, lab.label_rect)
                        if rect_hit:
                            out.append(("node-label", node.node_id, lab.child_id))
                    elif ka in ("n", "b") and kb in ("n", "b") and ka != kb:
                        pass
    # labels vs arestas alheias, e arestas atravessando nós alheios
    for e in labels:
        for other in labels:
            if other is e:
                continue
            if _segment_rect_overlap(other.p0, other.p1, e.label_rect):
                out.append(("edge-label", other.child_id, e.child_id))
    for e in labels:
        x0 = min(e.p0[0], e.p1[0]); x1 = max(e.p0[0], e.p1[0])
        y0 = min(e.p0[1], e.p1[1]); y1 = max(e.p0[1], e.p1[1])
        seg_cells = set(cells((x0, y0, x1, y1)))
        checked = set()
        for c in seg_cells:
            for kind, obj in grid.get(c, ()):
                if kind != "n" or obj.node_id in (e.parent_id, e.child_id) or obj.node_id in checked:
                    continue
                checked.add(obj.node_id)
                if _segment_ellipse_overlap(e.p0, e.p1, obj):
                    out.append(("edge-node", e.child_id, obj.node_id))
    return out


def compute_layout(
    model: TreeVisualizationModel,
    measure: Optional[TextMeasure] = None,
    *,
    collapsed: Optional[Set[int]] = None,
    max_depth: Optional[int] = None,
    min_samples: Optional[int] = None,
    show_uncertainty: bool = False,
    scientific: bool = False,
    show_ids: bool = False,
    preset: Optional[str] = None,
    max_iterations: int = 14,
) -> Layout:
    """Calcula posições + resolve colisões por espaçamento adaptativo (nunca remove nem move ao acaso)."""
    measure = measure or TextMeasure()
    visible, hidden_under = visible_set(model, set(collapsed or ()), max_depth, min_samples)
    name = preset or preset_for(len(visible))
    params = LayoutParams(preset=name, **PRESETS[name])
    hscale = vscale = 1.0
    layout = None
    for it in range(max_iterations + 1):
        layout = _build(model, measure, visible, hidden_under, params, hscale, vscale,
                        show_uncertainty, scientific, show_ids)
        layout.overlaps = find_overlaps(layout)
        layout.iterations = it
        if not layout.overlaps:
            break
        kinds = {o[0] for o in layout.overlaps}
        if kinds & {"node-node", "label-label", "edge-label"}:
            hscale *= 1.18
        if kinds & {"node-label", "edge-node", "edge-label"}:
            vscale *= 1.12
            hscale *= 1.06
    return layout


def diagnostics(model: TreeVisualizationModel, layout: Layout) -> dict:
    """TREE VISUALIZATION DIAGNOSTIC: compara nós lógicos vs renderizados e documenta o que está escondido."""
    return {
        "algorithm": model.algorithm,
        "logical_node_count": int(model.logical_node_count),
        "model_node_count": int(model.node_count),
        "rendered_node_count": int(layout.rendered_node_count),
        "hidden_node_count": int(layout.hidden_count),
        "rendered_plus_hidden": int(layout.rendered_node_count + layout.hidden_count),
        "consistent": bool(layout.rendered_node_count + layout.hidden_count == model.logical_node_count),
        "depth": int(model.depth),
        "layout": "hierarchical (Reingold-Tilford, subtree contours)",
        "preset": layout.params.preset,
        "bounding_box": [round(v, 2) for v in layout.bbox],
        "overlaps_detected": len(layout.overlaps),
        "spacing_iterations": int(layout.iterations),
        "hidden_or_collapsed": int(layout.hidden_count),
        "verdict": (
            "árvore realmente pequena" if model.logical_node_count <= 3 and layout.rendered_node_count == model.logical_node_count
            else ("BUG DE VISUALIZAÇÃO: nós lógicos != renderizados"
                  if layout.rendered_node_count + layout.hidden_count != model.logical_node_count else "ok")
        ),
    }
