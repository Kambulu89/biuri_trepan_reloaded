"""Renderer Qt (vectorial) de nós ELÍPTICOS + exportação PNG/SVG/PDF/JSON.

O mesmo ``draw_scene`` serve o widget e a exportação: a figura exportada é igual ao ecrã,
mas usa o layout COMPLETO (sem filtros/colapsos) e não inclui botões nem scrollbars.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Set, Tuple

from PyQt6.QtCore import QMarginsF, QPointF, QRectF, QSize, QSizeF, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QFontMetricsF, QImage, QPageSize, QPainter, QPdfWriter, QPen,
)

from gui.tree_viz.labels import class_color_map
from gui.tree_viz.layout import Layout, TextMeasure, compute_layout, diagnostics
from gui.tree_viz.model import TreeVisualizationModel, VizNode
from gui.tree_viz.strings import tr

FONT_FAMILY = "Sans Serif"
BG = QColor("#FAFAFA")
INTERNAL_FILL = QColor("#F1F4FA")
INTERNAL_BORDER = QColor("#2F3B52")
LEFT_EDGE = QColor(51, 130, 190)      # ramo ≤ / não (mantém o esquema original azul)
RIGHT_EDGE = QColor(204, 102, 51)     # ramo > / sim (mantém o esquema original laranja)
CHILD_EDGE = QColor(110, 110, 110)
ACCENT = QColor("#D81B60")
TEXT = QColor("#111111")


def _font(px: float, bold: bool = False) -> QFont:
    f = QFont(FONT_FAMILY)
    f.setPixelSize(max(1, int(round(px))))
    f.setBold(bold)
    return f


class QtTextMeasure(TextMeasure):
    def __init__(self):
        self._cache: Dict[tuple, float] = {}

    def width(self, text: str, px: float, bold: bool = False) -> float:
        key = (text, round(px, 2), bold)
        if key not in self._cache:
            self._cache[key] = QFontMetricsF(_font(px, bold)).horizontalAdvance(text)
        return self._cache[key]


@dataclass
class RenderOptions:
    show_uncertainty: bool = False
    scientific: bool = False
    show_ids: bool = False
    show_semantic: bool = True
    selected_id: Optional[int] = None
    selected_edge: Optional[Tuple[int, int]] = None
    path_ids: Set[int] = field(default_factory=set)
    hover_id: Optional[int] = None


def _edge_color(branch: str) -> QColor:
    return LEFT_EDGE if branch == "left" else RIGHT_EDGE if branch == "right" else CHILD_EDGE


def draw_scene(painter: QPainter, model: TreeVisualizationModel, layout: Layout, opts: RenderOptions,
               colors: Optional[Dict[str, Tuple[str, str]]] = None) -> None:
    """Desenha em coordenadas de mundo (o chamador aplica translate/scale do viewport)."""
    colors = colors or class_color_map(model.class_labels())
    p = layout.params
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
    path_edges = {(a, b) for a, b in zip(sorted(opts.path_ids, key=lambda i: model.nodes[i].depth),
                                         sorted(opts.path_ids, key=lambda i: model.nodes[i].depth)[1:])}
    # 1) arestas
    for e in layout.edges:
        edge = model.edge(e.parent_id, e.child_id)
        on_path = (e.parent_id, e.child_id) in path_edges
        sel = opts.selected_edge == (e.parent_id, e.child_id)
        pen = QPen(ACCENT if (on_path or sel) else _edge_color(edge.branch), 3.2 if (on_path or sel) else 1.8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.drawLine(QPointF(*e.p0), QPointF(*e.p1))
    # 2) labels das arestas (junto à aresta, fora dos nós)
    painter.setFont(_font(p.label_font_px))
    for e in layout.edges:
        r = QRectF(e.label_rect[0], e.label_rect[1], e.label_rect[2] - e.label_rect[0], e.label_rect[3] - e.label_rect[1])
        sel = opts.selected_edge == (e.parent_id, e.child_id)
        painter.setPen(QPen(ACCENT if sel else QColor(190, 190, 190), 1.2 if sel else 0.8))
        painter.setBrush(QColor(255, 255, 255, 235))
        painter.drawRoundedRect(r, 3, 3)
        painter.setPen(QPen(TEXT))
        painter.drawText(r, Qt.AlignmentFlag.AlignCenter, e.label)
    # 3) nós (elipses/círculos)
    for nid, sh in layout.nodes.items():
        n = model.nodes[nid]
        in_path = nid in opts.path_ids
        selected = opts.selected_id == nid
        if n.is_leaf:
            fill, txt = colors.get(str(n.class_prediction), ("#BBBBBB", "#111111"))
            painter.setBrush(QBrush(QColor(fill)))
            border = QColor(fill).darker(165)
            width = 1.6
        else:
            painter.setBrush(QBrush(INTERNAL_FILL))
            border, width, txt = INTERNAL_BORDER, 2.6, "#111111"
        if in_path:
            border, width = ACCENT.lighter(120), max(width, 3.0)
        if selected:
            border, width = ACCENT, 4.2
        if opts.hover_id == nid and not selected:
            width += 1.0
        painter.setPen(QPen(border, width))
        painter.drawEllipse(QRectF(sh.cx - sh.rx, sh.cy - sh.ry, 2 * sh.rx, 2 * sh.ry))
        if not n.is_leaf:  # marcador subtil: anel interno distingue interno de folha, mantendo o círculo
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor(47, 59, 82, 70), 1.0))
            painter.drawEllipse(QRectF(sh.cx - sh.rx + 4, sh.cy - sh.ry + 4, 2 * sh.rx - 8, 2 * sh.ry - 8))
        # texto compacto (nunca regras longas)
        painter.setPen(QPen(QColor(txt)))
        heights = [(p.font_px if i == 0 else p.small_font_px) * 1.22 for i in range(len(sh.lines))]
        y = sh.cy - sum(heights) / 2
        for i, line in enumerate(sh.lines):
            painter.setFont(_font(p.font_px if i == 0 else p.small_font_px, bold=(i == 0)))
            painter.drawText(QRectF(sh.cx - sh.rx, y, 2 * sh.rx, heights[i]), Qt.AlignmentFlag.AlignCenter, line)
            y += heights[i]
        # badge semântico (só com contribuição semântica concreta)
        if opts.show_semantic and sh.semantic_badge:
            bx, by, br = sh.semantic_badge
            painter.setPen(QPen(QColor("#7A4B00"), 1.2))
            painter.setBrush(QColor("#FFC107"))
            painter.drawEllipse(QPointF(bx, by), br, br)
            painter.setFont(_font(br * 1.3, bold=True))
            painter.setPen(QPen(QColor("#2B1A00")))
            painter.drawText(QRectF(bx - br, by - br, 2 * br, 2 * br), Qt.AlignmentFlag.AlignCenter, "S")
        # indicador de subárvore colapsada: "+ N nós"
        if sh.badge_rect:
            painter.setFont(_font(p.small_font_px))
            r = QRectF(sh.badge_rect[0], sh.badge_rect[1], sh.badge_rect[2] - sh.badge_rect[0], sh.badge_rect[3] - sh.badge_rect[1])
            painter.setPen(QPen(QColor("#555555"), 1.0, Qt.PenStyle.DashLine))
            painter.setBrush(QColor("#FFFFFF"))
            painter.drawRoundedRect(r, r.height() / 2, r.height() / 2)
            painter.setPen(QPen(QColor("#222222")))
            painter.drawText(r, Qt.AlignmentFlag.AlignCenter, tr("collapsed_more", n=sh.collapsed_hidden))


def legend_items(model: TreeVisualizationModel, colors) -> list:
    classes = [c for c in model.class_labels() if any(n.is_leaf and str(n.class_prediction) == c for n in model.nodes.values())]
    items = [("internal", tr("legend_internal"), None)]
    items += [("leaf", f"{tr('legend_leaf')} {c}", colors.get(c, ("#BBBBBB", "#111111"))[0]) for c in classes]
    if any(n.split_type == "m_of_n" for n in model.nodes.values()):
        items.append(("text", tr("legend_mofn"), None))
    if any(n.is_semantic_split for n in model.nodes.values()):
        items.append(("badge", tr("legend_semantic"), None))
    if any(n.is_ontology_feature for n in model.nodes.values()):
        items.append(("text", tr("legend_onto"), None))
    return items


def draw_legend(painter: QPainter, model: TreeVisualizationModel, colors, x: float, y: float, px: float = 12.0) -> QSizeF:
    items = legend_items(model, colors)
    painter.setFont(_font(px))
    fm = QFontMetricsF(_font(px))
    w = max(fm.horizontalAdvance(t) for _, t, _ in items) + px * 2.6
    h = len(items) * px * 1.55 + px * 0.6
    painter.setPen(QPen(QColor(200, 200, 200), 0.8))
    painter.setBrush(QColor(255, 255, 255, 225))
    painter.drawRoundedRect(QRectF(x, y, w, h), 4, 4)
    cy = y + px * 1.0
    for kind, text, fill in items:
        r = px * 0.45
        cx = x + px * 0.9
        if kind == "internal":
            painter.setBrush(INTERNAL_FILL); painter.setPen(QPen(INTERNAL_BORDER, 1.8))
            painter.drawEllipse(QPointF(cx, cy), r, r)
        elif kind == "leaf":
            painter.setBrush(QColor(fill)); painter.setPen(QPen(QColor(fill).darker(165), 1.4))
            painter.drawEllipse(QPointF(cx, cy), r, r)
        elif kind == "badge":
            painter.setBrush(QColor("#FFC107")); painter.setPen(QPen(QColor("#7A4B00"), 1))
            painter.drawEllipse(QPointF(cx, cy), r, r)
        painter.setPen(QPen(TEXT))
        painter.drawText(QPointF(x + px * 1.9, cy + px * 0.35), text)
        cy += px * 1.55
    return QSizeF(w, h)


# ================================================================================ export
def default_dpi_scale(width_world: float, min_px: float = 2400.0, max_pixels: float = 1.2e8) -> float:
    return max(1.0, min_px / max(1.0, width_world))


def export_tree(model: TreeVisualizationModel, path: str, *, fmt: Optional[str] = None, title: Optional[str] = None,
                scientific: bool = False, show_ids: bool = False, show_legend: bool = True,
                metadata: Optional[dict] = None, write_json: bool = False) -> dict:
    """Exporta a árvore COMPLETA (sem filtros) para PNG/SVG/PDF. Não inclui botões/painéis da GUI."""
    fmt = (fmt or Path(path).suffix.lstrip(".") or "png").lower()
    measure = QtTextMeasure()
    layout = compute_layout(model, measure, scientific=scientific, show_ids=show_ids)
    colors = class_color_map(model.class_labels())
    margin = 28.0
    x0, y0, x1, y1 = layout.bbox
    tree_w, tree_h = x1 - x0, y1 - y0
    title_h = 26.0 if title else 0.0
    legend_px = 12.0
    # legenda por baixo da árvore (alinhada à esquerda), fora da área da árvore
    n_legend = len(legend_items(model, colors)) if show_legend else 0
    legend_h = (n_legend * legend_px * 1.55 + legend_px * 0.6 + 8) if show_legend else 0.0
    legend_w = 230.0 if show_legend else 0.0
    W = max(tree_w, legend_w) + 2 * margin
    H = tree_h + 2 * margin + title_h + legend_h
    ox = margin + (max(tree_w, legend_w) - tree_w) / 2 - x0
    oy = margin + title_h - y0

    def paint(painter: QPainter):
        painter.fillRect(QRectF(0, 0, W, H), QColor("white"))
        if title:
            painter.setFont(_font(14, bold=True))
            painter.setPen(QPen(QColor("#222222")))
            painter.drawText(QRectF(margin, 6, W - 2 * margin, title_h), Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, title)
        painter.save()
        painter.translate(ox, oy)
        draw_scene(painter, model, layout, RenderOptions(scientific=scientific, show_ids=show_ids), colors)
        painter.restore()
        if show_legend:
            draw_legend(painter, model, colors, margin, margin + title_h + tree_h + 14, legend_px)

    out_path = str(Path(path).with_suffix("." + fmt))
    info = {"path": out_path, "format": fmt, "width_world": W, "height_world": H}
    if fmt == "png":
        scale = default_dpi_scale(W)
        while W * scale * H * scale > 1.2e8 and scale > 1.0:
            scale *= 0.8
        img = QImage(int(math.ceil(W * scale)), int(math.ceil(H * scale)), QImage.Format.Format_ARGB32)
        img.fill(QColor("white"))
        painter = QPainter(img)
        painter.scale(scale, scale)
        paint(painter)
        painter.end()
        img.setDotsPerMeterX(int(300 / 0.0254)); img.setDotsPerMeterY(int(300 / 0.0254))
        if not img.save(out_path, "PNG"):
            raise OSError(f"não foi possível guardar {out_path}")
        info.update(pixel_width=img.width(), pixel_height=img.height(), scale=scale)
    elif fmt == "svg":
        from PyQt6.QtSvg import QSvgGenerator
        gen = QSvgGenerator()
        gen.setFileName(out_path)
        gen.setSize(QSize(int(math.ceil(W)), int(math.ceil(H))))
        gen.setViewBox(QRectF(0, 0, W, H))
        gen.setTitle(title or model.algorithm)
        painter = QPainter(gen)
        paint(painter)
        painter.end()
    elif fmt == "pdf":
        pdf = QPdfWriter(out_path)
        pdf.setResolution(72)
        pdf.setPageSize(QPageSize(QSizeF(W, H), QPageSize.Unit.Point))
        pdf.setPageMargins(QMarginsF(0, 0, 0, 0))
        painter = QPainter(pdf)
        paint(painter)
        painter.end()
    else:
        raise ValueError(f"formato de exportação não suportado: {fmt}")
    diag = diagnostics(model, layout)
    info["diagnostic"] = diag
    if write_json:
        jp = str(Path(out_path).with_suffix(".json"))
        Path(jp).write_text(json.dumps({"tree": model.to_dict(), "diagnostic": diag, "metadata": metadata or {}},
                                       indent=2, default=str, ensure_ascii=False), encoding="utf-8")
        info["json"] = jp
    return info
