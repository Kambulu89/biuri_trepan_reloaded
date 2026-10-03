"""Widget interactivo de árvores (nós ELÍPTICOS) sobre o pacote ``gui.tree_viz``.

Camadas: árvore científica (só leitura) -> TreeVisualizationModel -> layout hierárquico ->
renderer -> viewport (zoom/pan). Zoom/pan NUNCA recalculam o layout; filtros/colapsos são
puramente visuais (nada é podado nem alterado na árvore).
"""
import math
from typing import Optional

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import QApplication, QTextEdit, QToolTip, QVBoxLayout, QWidget

from gui.tree_viz import details as _details
from gui.tree_viz.labels import class_color_map
from gui.tree_viz.layout import compute_layout, diagnostics
from gui.tree_viz.model import (
    TreeVisualizationError, TreeVisualizationModel, VizEdge, VizNode,
    build_visualization_model, tree_signature,
)
from gui.tree_viz.render import QtTextMeasure, RenderOptions, draw_legend, draw_scene, export_tree
from gui.tree_viz.strings import tr

MIN_ZOOM, MAX_ZOOM = 0.08, 6.0


class TreeNode:
    
    def __init__(self, node_id, feature=None, threshold=None, left=None, right=None, 
                 samples=None, values=None, class_name=None, is_leaf=False,
                 condition_text=None, false_label="≤", true_label=">"):
        self.node_id = node_id
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.samples = samples
        self.values = values
        self.class_name = class_name
        self.is_leaf = is_leaf
        self.condition_text = condition_text
        self.false_label = false_label
        self.true_label = true_label
        self.x = 0
        self.y = 0
        self.width = 80
        self.height = 40
        self.uncertainty = self._calculate_uncertainty()
        
    def _calculate_uncertainty(self):
        
        if self.values is None or len(self.values) == 0:
            return 0.0
        
        total_samples = sum(self.values)
        if total_samples == 0:
            return 1.0
        
        # Calcula entropia
        probabilities = [v / total_samples for v in self.values if v > 0]
        entropy = -sum(p * math.log2(p) for p in probabilities)
        
        # Normaliza para [0, 1]
        max_entropy = math.log2(len(self.values))
        return entropy / max_entropy if max_entropy > 0 else 0.0



class TreeDetailsPanel(QTextEdit):
    """Painel de detalhes (nó/aresta) ligado ao widget da árvore: detalhes sob demanda."""

    def __init__(self, tree_widget=None, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Monospace", 9))
        self.setMinimumWidth(260)
        self.setPlaceholderText(tr("node_details"))
        self._tree_widget = None
        if tree_widget is not None:
            self.attach(tree_widget)

    def attach(self, tree_widget):
        self._tree_widget = tree_widget
        tree_widget.selection_changed.connect(self.show_selection)
        self.show_selection(tree_widget.selected_object())

    def show_selection(self, obj):
        w = self._tree_widget
        if obj is None or w is None or w.model is None:
            self.clear()
        elif isinstance(obj, VizEdge):
            self.setPlainText(_details.edge_details_text(w.model, obj.parent_id, obj.child_id))
        else:
            self.setPlainText(_details.node_details_text(w.model, obj.node_id))


class InteractiveTreeWidget(QWidget):

    node_clicked = pyqtSignal(object)        # VizNode (compatível com o sinal antigo)
    selection_changed = pyqtSignal(object)   # VizNode | VizEdge | None
    view_changed = pyqtSignal()

    def __init__(self, tree_model, feature_names, class_names, parent=None, algorithm=None):
        super().__init__(parent)
        self.tree_model = tree_model
        self.algorithm = algorithm
        self.feature_names = self._align_feature_names(feature_names, tree_model)
        self.class_names = list(class_names or [])
        self._export_class_names = self._class_names_for_tree(tree_model, self.class_names)
        self._measure = QtTextMeasure()
        self.model: Optional[TreeVisualizationModel] = None
        self.layout = None
        self.error: Optional[str] = None
        self._layout_cache = {}
        # viewport (única transformação: screen = world * zoom + offset)
        self.zoom_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        # opções visuais (nunca tocam na árvore)
        self.show_uncertainty = False
        self.scientific_view = False
        self.show_node_ids = False
        self.show_semantic = True
        self.show_legend = True
        self.min_samples_threshold = None
        self.max_depth_display = None
        self.collapsed = set()
        self.selected_id = None
        self.selected_edge = None
        self.highlight_path = True
        self.hovered_node = None
        self.dragging = False
        self._press_pos = None
        self.setMinimumSize(520, 380)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._rebuild_model()
        self._relayout()
        self.initial_view()

    # ---------------------------------------------------------------- compat helpers
    @staticmethod
    def _align_feature_names(feature_names, tree_model):
        """Garante len(feature_names) >= n_features da árvore (Reloaded pode ter onto_*)."""
        n_features = 0
        if tree_model is not None and hasattr(tree_model, 'tree_'):
            n_features = int(getattr(tree_model.tree_, 'n_features', 0))
        elif tree_model is not None:
            n_features = int(getattr(tree_model, 'n_features_in_', 0) or 0)
        names = list(feature_names or [])
        if n_features <= 0:
            return names
        if len(names) >= n_features:
            return names[:n_features]
        names.extend([f"feature_{i}" for i in range(len(names), n_features)])
        return names

    @staticmethod
    def _class_names_for_tree(tree_model, class_names):
        if tree_model is None:
            return list(class_names or [])
        classes = np.asarray(getattr(tree_model, 'classes_', []))
        if classes.size == 0:
            return list(class_names or [])
        names = list(class_names or [])
        out = []
        for c in classes:
            try:
                ci = int(c)
            except (TypeError, ValueError):
                out.append(str(c))
                continue
            out.append(str(names[ci]) if 0 <= ci < len(names) else str(c))
        return out

    # ---------------------------------------------------------------- model / layout
    def _build_tree_structure(self):
        """Constrói o modelo visual a partir da árvore (TREPAN nativo, C4.5 nativo ou sklearn-like)."""
        if self.tree_model is None:
            raise TreeVisualizationError("NO_TREE", tr("no_tree"))
        if hasattr(self.tree_model, 'root_') and hasattr(self.tree_model, 'export_rules'):
            return self._build_historical_trepan_structure()
        return build_visualization_model(self.tree_model, self.feature_names, self.class_names, self.algorithm)

    def _build_historical_trepan_structure(self):
        return build_visualization_model(self.tree_model, self.feature_names, self.class_names, self.algorithm)

    def _rebuild_model(self):
        self.error = None
        self.model = None
        try:
            self.model = self._build_tree_structure()
        except TreeVisualizationError as exc:
            self.error = str(exc)
        except Exception as exc:  # nunca crashar: mostrar diagnóstico
            self.error = f"{tr('viz_error')} [{type(exc).__name__}] {exc}"
        self._layout_cache.clear()

    def _layout_key(self):
        return (id(self.model), frozenset(self.collapsed), self.max_depth_display, self.min_samples_threshold,
                self.show_uncertainty, self.scientific_view, self.show_node_ids)

    def _relayout(self):
        """Recalcula o layout (só quando estrutura/labels/filtros mudam; zoom/pan não passam por aqui)."""
        if self.model is None:
            self.layout = None
            return
        key = self._layout_key()
        if key not in self._layout_cache:
            if len(self._layout_cache) > 8:
                self._layout_cache.clear()
            self._layout_cache[key] = compute_layout(
                self.model, self._measure, collapsed=set(self.collapsed), max_depth=self.max_depth_display,
                min_samples=self.min_samples_threshold, show_uncertainty=self.show_uncertainty,
                scientific=self.scientific_view, show_ids=self.show_node_ids)
        self.layout = self._layout_cache[key]

    def update_tree(self, tree_model, feature_names=None, class_names=None, algorithm=None):
        """Troca a árvore: NÃO reconstrói nenhum modelo; recalcula layout e ajusta a vista."""
        self.tree_model = tree_model
        if algorithm is not None:
            self.algorithm = algorithm
        if class_names is not None:
            self.class_names = list(class_names)
        self.feature_names = self._align_feature_names(
            feature_names if feature_names is not None else self.feature_names, tree_model)
        self._export_class_names = self._class_names_for_tree(tree_model, self.class_names)
        self.collapsed.clear(); self.selected_id = None; self.selected_edge = None
        self.hovered_node = None; self.min_samples_threshold = None; self.max_depth_display = None
        self._rebuild_model()
        self._relayout()
        self.initial_view()
        self.selection_changed.emit(None)
        self.update()

    # ---------------------------------------------------------------- viewport
    def _world_bbox(self):
        return self.layout.bbox if self.layout else (0, 0, 1, 1)

    def fit_tree_to_view(self, legible: bool = False, margin: float = 28.0):
        """Ajusta a bounding box REAL ao viewport (sem constantes fixas). legible=True aplica piso de legibilidade."""
        if self.layout is None or self.width() < 10 or self.height() < 10:
            return
        x0, y0, x1, y1 = self._world_bbox()
        bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
        zoom = min((self.width() - 2 * margin) / bw, (self.height() - 2 * margin) / bh)
        zoom = min(zoom, self.layout.params.initial_zoom_cap)
        floor_applied = False
        if legible and zoom < self.layout.params.min_legible_zoom:
            zoom, floor_applied = self.layout.params.min_legible_zoom, True
        self.zoom_factor = max(MIN_ZOOM, min(MAX_ZOOM, zoom))
        cx = (x0 + x1) / 2
        self.offset_x = self.width() / 2 - cx * self.zoom_factor
        if floor_applied:  # árvore grande: topo da árvore visível, resto por pan
            self.offset_y = margin - y0 * self.zoom_factor
        else:
            self.offset_y = self.height() / 2 - (y0 + y1) / 2 * self.zoom_factor
        self.view_changed.emit(); self.update()

    def initial_view(self):
        self.fit_tree_to_view(legible=True)

    def reset_zoom(self):
        self.fit_tree_to_view(legible=False)

    def reset_view(self):
        """Restaura zoom, pan e selecção sem reconstruir a árvore."""
        self.selected_id = None; self.selected_edge = None
        self.selection_changed.emit(None)
        self.initial_view()

    def center_on_node(self, node_id):
        if self.layout is None or node_id not in self.layout.nodes:
            return
        sh = self.layout.nodes[node_id]
        self.offset_x = self.width() / 2 - sh.cx * self.zoom_factor
        self.offset_y = self.height() / 2 - sh.cy * self.zoom_factor
        self.view_changed.emit(); self.update()

    def center_root(self):
        if self.model is not None:
            sh = self.layout.nodes[self.model.root_id]
            self.offset_x = self.width() / 2 - sh.cx * self.zoom_factor
            self.offset_y = 40 - (sh.cy - sh.ry) * self.zoom_factor
            self.view_changed.emit(); self.update()

    def center_selected(self):
        if self.selected_id is not None:
            self.center_on_node(self.selected_id)

    def center_tree(self):  # compat: "Centrar" = ajustar à vista
        self.fit_tree_to_view(legible=False)

    def _zoom_about(self, factor, sx, sy):
        new = max(MIN_ZOOM, min(MAX_ZOOM, self.zoom_factor * factor))
        if new == self.zoom_factor:
            return
        wx, wy = (sx - self.offset_x) / self.zoom_factor, (sy - self.offset_y) / self.zoom_factor
        self.zoom_factor = new
        self.offset_x, self.offset_y = sx - wx * new, sy - wy * new
        self.view_changed.emit(); self.update()

    def zoom_in(self):
        self._zoom_about(1.2, self.width() / 2, self.height() / 2)

    def zoom_out(self):
        self._zoom_about(1 / 1.2, self.width() / 2, self.height() / 2)

    def to_world(self, sx, sy):
        return (sx - self.offset_x) / self.zoom_factor, (sy - self.offset_y) / self.zoom_factor

    # ---------------------------------------------------------------- visual options (never touch the tree)
    def toggle_uncertainty_display(self):
        self.show_uncertainty = not self.show_uncertainty
        self._relayout(); self.update()

    def set_scientific_view(self, on: bool):
        self.scientific_view = bool(on); self._relayout(); self.update()

    def set_show_node_ids(self, on: bool):
        self.show_node_ids = bool(on); self._relayout(); self.update()

    def set_complexity_filter(self, min_samples, max_depth):
        """Filtro SÓ VISUAL (esconde/colapsa; indicado como '+N nós'). Não executa pruning."""
        self.min_samples_threshold = min_samples if (min_samples or 0) > 0 else None
        self.max_depth_display = max_depth if (max_depth is not None and max_depth >= 0) else None
        self._relayout(); self.update()

    def clear_filters(self):
        self.min_samples_threshold = None; self.max_depth_display = None; self.collapsed.clear()
        self._relayout(); self.update()

    def toggle_collapse(self, node_id=None):
        nid = self.selected_id if node_id is None else node_id
        if nid is None or self.model is None or not self.model.nodes[nid].children:
            return
        self.collapsed.symmetric_difference_update({nid})
        self._relayout(); self.update()

    def expand_all(self):
        self.collapsed.clear(); self._relayout(); self.update()

    def search(self, text):
        """Nós cuja feature (completa), classe, id ou condição contém ``text``."""
        if self.model is None or not str(text).strip():
            return []
        q = str(text).strip().lower()
        out = []
        for nid in self.model.preorder():
            n = self.model.nodes[nid]
            hay = [str(n.node_id), (n.feature_full or ''), (n.class_prediction or '')] + [c['text_full'] for c in n.conditions]
            if any(q in h.lower() for h in hay):
                out.append(nid)
        return out

    def _reveal(self, node_id):
        """Expande ancestrais colapsados para tornar o nó visível (só visual)."""
        changed = False
        for anc in self.model.path_to(node_id)[:-1]:
            if anc in self.collapsed:
                self.collapsed.discard(anc); changed = True
        if self.max_depth_display is not None and self.model.nodes[node_id].depth > self.max_depth_display:
            self.max_depth_display = None; changed = True
        if self.min_samples_threshold is not None:
            self.min_samples_threshold = None; changed = True
        if changed:
            self._relayout()

    def select_node(self, node_id, center=False):
        if self.model is None or node_id not in self.model.nodes:
            return
        self._reveal(node_id)
        self.selected_id, self.selected_edge = node_id, None
        node = self.model.nodes[node_id]
        if center:
            self.center_on_node(node_id)
        self.node_clicked.emit(node)
        self.selection_changed.emit(node)
        self.update()

    def selected_object(self):
        if self.model is None:
            return None
        if self.selected_edge is not None:
            return self.model.edge(*self.selected_edge)
        return self.model.nodes.get(self.selected_id) if self.selected_id is not None else None

    def path_ids(self):
        if not self.highlight_path or self.model is None:
            return set()
        if self.selected_id is not None:
            return set(self.model.path_to(self.selected_id))
        if self.selected_edge is not None:
            return set(self.model.path_to(self.selected_edge[1]))
        return set()

    def toggle_highlight_path(self):
        self.highlight_path = not self.highlight_path; self.update()

    # ---------------------------------------------------------------- diagnostics / export
    def diagnostic(self):
        if self.model is None or self.layout is None:
            return {"error": self.error}
        d = diagnostics(self.model, self.layout)
        d["zoom"] = round(self.zoom_factor, 3)
        d["selected_tree_signature"] = tree_signature(self.tree_model) if self.tree_model is not None else None
        return d

    def hidden_notice(self):
        if self.layout is None or not self.layout.hidden_count:
            return ""
        return tr("hidden_notice", shown=self.layout.rendered_node_count, total=self.model.logical_node_count,
                  hidden=self.layout.hidden_count)

    def export(self, path, fmt=None, title=None, write_json=False, metadata=None):
        if self.model is None:
            raise TreeVisualizationError("NO_TREE", self.error or tr("no_tree"))
        return export_tree(self.model, path, fmt=fmt, title=title, scientific=self.scientific_view,
                           show_ids=self.show_node_ids, metadata=metadata, write_json=write_json)

    # ---------------------------------------------------------------- painting
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#FAFAFA"))
        if self.model is None or self.layout is None:
            painter.setPen(QPen(QColor("#444444")))
            painter.setFont(QFont("Sans Serif", 11))
            msg = f"{tr('no_tree')}\n{tr('reason')}: {self.error or tr('reason_untrained')}"
            painter.drawText(self.rect().adjusted(24, 24, -24, -24), Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, msg)
            painter.end()
            return
        colors = class_color_map(self.model.class_labels())
        opts = RenderOptions(self.show_uncertainty, self.scientific_view, self.show_node_ids, self.show_semantic,
                             self.selected_id, self.selected_edge, self.path_ids(), self.hovered_node)
        painter.save()
        painter.translate(self.offset_x, self.offset_y)
        painter.scale(self.zoom_factor, self.zoom_factor)
        draw_scene(painter, self.model, self.layout, opts, colors)
        painter.restore()
        if self.show_legend:
            draw_legend(painter, self.model, colors, 10, 10, 11.0)
        notice = self.hidden_notice()
        if notice:
            painter.setPen(QPen(QColor("#8A3B00")))
            painter.setFont(QFont("Sans Serif", 9))
            painter.drawText(self.rect().adjusted(10, 0, -10, -8), Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight, notice)
        painter.end()

    # ---------------------------------------------------------------- interaction
    def _node_at(self, sx, sy):
        if self.layout is None:
            return None
        wx, wy = self.to_world(sx, sy)
        for nid, sh in self.layout.nodes.items():
            if ((wx - sh.cx) / sh.rx) ** 2 + ((wy - sh.cy) / sh.ry) ** 2 <= 1.0:
                return nid
        return None

    def _edge_at(self, sx, sy):
        if self.layout is None:
            return None
        wx, wy = self.to_world(sx, sy)
        tol = 6.0 / self.zoom_factor
        for e in self.layout.edges:
            r = e.label_rect
            if r[0] <= wx <= r[2] and r[1] <= wy <= r[3]:
                return (e.parent_id, e.child_id)
            (ax, ay), (bx, by) = e.p0, e.p1
            dx, dy = bx - ax, by - ay
            L2 = dx * dx + dy * dy
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((wx - ax) * dx + (wy - ay) * dy) / L2))
            if math.hypot(wx - (ax + t * dx), wy - (ay + t * dy)) <= tol:
                return (e.parent_id, e.child_id)
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            self._press_pos = (pos.x(), pos.y())
            self.dragging = True
            self._last = (pos.x(), pos.y())
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        pos = event.position()
        if self.dragging:
            self.offset_x += pos.x() - self._last[0]
            self.offset_y += pos.y() - self._last[1]
            self._last = (pos.x(), pos.y())
            self.update()
        else:
            nid = self._node_at(pos.x(), pos.y())
            if nid != self.hovered_node:
                self.hovered_node = nid
                self.update()
            if nid is not None and self.model is not None:
                QToolTip.showText(event.globalPosition().toPoint(), _details.node_tooltip(self.model, nid), self)
            else:
                QToolTip.hideText()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            pos = event.position()
            moved = self._press_pos is not None and math.hypot(pos.x() - self._press_pos[0], pos.y() - self._press_pos[1]) > 4
            self.dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            if not moved and self.model is not None:
                nid = self._node_at(pos.x(), pos.y())
                if nid is not None:
                    self.select_node(nid)
                else:
                    edge = self._edge_at(pos.x(), pos.y())
                    if edge is not None:
                        self.selected_edge, self.selected_id = edge, None
                        self.selection_changed.emit(self.model.edge(*edge)); self.update()
                    elif self.selected_id is not None or self.selected_edge is not None:
                        self.selected_id = self.selected_edge = None
                        self.selection_changed.emit(None); self.update()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        nid = self._node_at(event.position().x(), event.position().y())
        if nid is not None:
            self.toggle_collapse(nid)
        super().mouseDoubleClickEvent(event)

    def wheelEvent(self, event):
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self._zoom_about(factor, event.position().x(), event.position().y())

    def resizeEvent(self, event):
        old = (event.oldSize().width(), event.oldSize().height())
        super().resizeEvent(event)
        if old[0] <= 0 or old[1] <= 0:
            self.initial_view()
        else:  # mantém o centro da vista ao redimensionar
            self.offset_x += (self.width() - old[0]) / 2
            self.offset_y += (self.height() - old[1]) / 2
