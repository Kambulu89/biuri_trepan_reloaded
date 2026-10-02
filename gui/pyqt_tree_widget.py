import sys
import math
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QScrollArea, QFrame, QSlider, QCheckBox, QDialog, QDialogButtonBox,
    QMessageBox, QApplication
)
from PyQt6.QtCore import Qt, QRect, QPoint, QSize, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QFont, QFontMetrics,
    QLinearGradient, QPainterPath, QPolygonF
)


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


class InteractiveTreeWidget(QWidget):
    
    
    node_clicked = pyqtSignal(object)  # Sinal emitido quando um nó é clicado
    
    def __init__(self, tree_model, feature_names, class_names, parent=None):
        super().__init__(parent)
        self.tree_model = tree_model
        self.feature_names = self._align_feature_names(feature_names, tree_model)
        self.class_names = list(class_names or [])
        self._export_class_names = self._class_names_for_tree(tree_model, self.class_names)
        
        # Configurações de visualização
        self.node_radius = 20
        self.level_height = 60
        self.min_node_spacing = 80
        self.zoom_factor = 1.0
        self.pan_x = 0
        self.pan_y = 0
        
        # Controles de complexidade
        self.show_uncertainty = True
        self.min_samples_threshold = 10
        self.max_depth_display = 5
        
        # Estado da interação
        self.hovered_node = None
        self.selected_path = []
        self.dragging = False
        self.last_pan_x = 0
        self.last_pan_y = 0
        
        # Controle de labels das arestas
        self.edge_labels = []
        self.label_positions = []
        
        # Dimensões da árvore
        self.tree_width = 0
        self.tree_height = 0
        self.tree_min_x = 0
        self.tree_max_x = 0
        self.tree_min_y = 0
        self.tree_max_y = 0
        
        # Constrói a estrutura da árvore
        self.tree_nodes = self._build_tree_structure()
        self._layout_tree()
        self._calculate_tree_bounds()
        self._auto_fit_tree()
        
        # Configurar widget
        self.setMinimumSize(800, 600)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)

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

    def _feature_name_at(self, feature_idx):
        if feature_idx is None or feature_idx < 0:
            return "?"
        idx = int(feature_idx)
        if idx < len(self.feature_names):
            return str(self.feature_names[idx])
        return f"feature_{idx}"

    def _leaf_class_name(self, values):
        if values is None or len(values) == 0 or self.tree_model is None:
            return None
        tree_classes = np.asarray(getattr(self.tree_model, 'classes_', []))
        if tree_classes.size == 0:
            idx = int(np.argmax(values))
            return (
                self.class_names[idx]
                if idx < len(self.class_names)
                else f"class_{idx}"
            )
        class_idx = int(np.argmax(values))
        if class_idx >= len(tree_classes):
            return f"class_{class_idx}"
        raw_label = tree_classes[class_idx]
        try:
            label = int(raw_label)
        except (TypeError, ValueError):
            return str(raw_label)
        if 0 <= label < len(self.class_names):
            return str(self.class_names[label])
        export = self._export_class_names
        if class_idx < len(export):
            return str(export[class_idx])
        return str(raw_label)

    def update_tree(self, tree_model, feature_names=None, class_names=None):
        """Troca árvore e nomes (ex.: Original 30 cols vs Reloaded 212 cols)."""
        self.tree_model = tree_model
        if class_names is not None:
            self.class_names = list(class_names)
        self.feature_names = self._align_feature_names(
            feature_names if feature_names is not None else self.feature_names,
            tree_model,
        )
        self._export_class_names = self._class_names_for_tree(tree_model, self.class_names)
        self.tree_nodes = self._build_tree_structure()
        self._layout_tree()
        self._calculate_tree_bounds()
        self._auto_fit_tree()
        self.update()
        
    def _build_tree_structure(self):
    
        if self.tree_model is None:
            return {}

        if hasattr(self.tree_model, 'root_') and hasattr(self.tree_model, 'export_rules'):
            return self._build_historical_trepan_structure()

        if not hasattr(self.tree_model, 'tree_'):
            raise TypeError('O visualizador recebeu um modelo que não expõe uma estrutura de árvore suportada.')

        tree = self.tree_model.tree_
        nodes = {}
        
        def build_node(node_id):
            if node_id == -1:
                return None
                
            feature = tree.feature[node_id]
            threshold = tree.threshold[node_id]
            samples = tree.n_node_samples[node_id]
            values = tree.value[node_id][0] if tree.value[node_id].size > 0 else []
            
            # Determina se é folha
            is_leaf = (tree.children_left[node_id] == -1 and 
                      tree.children_right[node_id] == -1)
            
            # Determina classe para folhas
            class_name = None
            if is_leaf and len(values) > 0:
                class_name = self._leaf_class_name(values)
            
            node = TreeNode(
                node_id=node_id,
                feature=feature,
                threshold=threshold,
                samples=samples,
                values=values,
                class_name=class_name,
                is_leaf=is_leaf
            )
            
            nodes[node_id] = node
            
            # Recursivamente constrói filhos
            if not is_leaf:
                left_child = build_node(tree.children_left[node_id])
                right_child = build_node(tree.children_right[node_id])
                node.left = left_child
                node.right = right_child
                
                if left_child:
                    nodes[tree.children_left[node_id]] = left_child
                if right_child:
                    nodes[tree.children_right[node_id]] = right_child
            
            return node
        
        root = build_node(0)
        return nodes

    def _build_historical_trepan_structure(self):
        nodes = {}
        model = self.tree_model

        def class_name_for_prediction(prediction):
            classes = np.asarray(getattr(model, 'classes_', []))
            matches = np.where(classes == prediction)[0] if classes.size else []
            if len(matches):
                idx = int(matches[0])
                if idx < len(self._export_class_names):
                    return str(self._export_class_names[idx])
            return str(prediction)

        def build(source):
            distribution = np.asarray(source.distribution, dtype=float)
            samples = int(len(source.real_y))
            values = distribution * max(1, samples)
            condition_text = None if source.is_leaf else source.test.text(self.feature_names)
            feature = None
            threshold = None
            if not source.is_leaf and source.test.literals:
                feature = int(source.test.literals[0].feature)
                threshold = float(source.test.literals[0].threshold)
            node = TreeNode(
                node_id=int(source.node_id),
                feature=feature,
                threshold=threshold,
                samples=samples,
                values=values,
                class_name=(class_name_for_prediction(source.prediction) if source.is_leaf else None),
                is_leaf=bool(source.is_leaf),
                condition_text=condition_text,
                false_label='não',
                true_label='sim',
            )
            nodes[node.node_id] = node
            if not source.is_leaf:
                node.left = build(source.false_child)
                node.right = build(source.true_child)
            return node

        build(model.root_)
        return nodes
    
    def _layout_tree(self):
        
        if not self.tree_nodes:
            return
            
        root = self.tree_nodes[0]
        
        max_depth = self._get_max_depth(root)
        
        self._calculate_positions(root, 0, max_depth)
        
    def _get_max_depth(self, node, depth=0):
        
        if node.is_leaf:
            return depth
        
        left_depth = self._get_max_depth(node.left, depth + 1) if node.left else depth
        right_depth = self._get_max_depth(node.right, depth + 1) if node.right else depth
        
        return max(left_depth, right_depth)
    
    def _calculate_positions(self, node, level, max_depth):
        
        if node.is_leaf:
            node.x = 0
            node.y = level * self.level_height
            return 0
        
        left_modifier = self._calculate_positions(node.left, level + 1, max_depth)
        right_modifier = self._calculate_positions(node.right, level + 1, max_depth)
        
        node.x = (node.left.x + node.right.x) / 2
        node.y = level * self.level_height
        
        separation = self.min_node_spacing
        if node.right.x - node.left.x < separation:
            shift = (separation - (node.right.x - node.left.x)) / 2
            self._shift_subtree(node.left, -shift)
            self._shift_subtree(node.right, shift)
        
        return left_modifier
    
    def _shift_subtree(self, node, shift):
        
        if node:
            node.x += shift
            self._shift_subtree(node.left, shift)
            self._shift_subtree(node.right, shift)
    
    def _calculate_tree_bounds(self):
        
        if not self.tree_nodes:
            return
            
        min_x = min(node.x for node in self.tree_nodes.values())
        max_x = max(node.x for node in self.tree_nodes.values())
        min_y = min(node.y for node in self.tree_nodes.values())
        max_y = max(node.y for node in self.tree_nodes.values())
        
        margin_x = self.node_radius * 4  
        margin_y = self.node_radius * 3  
        
        self.tree_min_x = min_x - margin_x
        self.tree_max_x = max_x + margin_x
        self.tree_min_y = min_y - margin_y
        self.tree_max_y = max_y + margin_y
        
        self.tree_width = self.tree_max_x - self.tree_min_x
        self.tree_height = self.tree_max_y - self.tree_min_y
    
    def _auto_fit_tree(self):
        
        if not self.tree_nodes or self.tree_width == 0 or self.tree_height == 0:
            return
            
        margin_factor = 0.85  
        zoom_x = (self.width() * margin_factor) / self.tree_width
        zoom_y = (self.height() * margin_factor) / self.tree_height
        
        # Usa o menor zoom para garantir que toda a árvore caiba
        self.zoom_factor = min(zoom_x, zoom_y, 1.0)  # Não aumenta além de 1.0
        
        # Garante zoom mínimo para legibilidade
        self.zoom_factor = max(self.zoom_factor, 0.1)
        
        # Centraliza a árvore perfeitamente
        center_x = (self.tree_min_x + self.tree_max_x) / 2
        center_y = (self.tree_min_y + self.tree_max_y) / 2
        
        # Calcula posição para centralizar considerando o zoom
        self.pan_x = (self.width() / 2) / self.zoom_factor - center_x
        self.pan_y = (self.height() / 2) / self.zoom_factor - center_y
        
        self.update()
    
    def paintEvent(self, event):
        
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Fundo
        painter.fillRect(self.rect(), QColor(240, 240, 240))
        
        if not self.tree_nodes:
            return
        
        # Desenha conexões primeiro
        self._draw_connections(painter)
        
        # Desenha nós
        self._draw_nodes(painter)
    
    def _draw_connections(self, painter):
        
        for node in self.tree_nodes.values():
            if not node.is_leaf:
                # Linha para filho esquerdo
                if node.left:
                    self._draw_connection(painter, node, node.left, node.false_label)
                
                # Linha para filho direito
                if node.right:
                    self._draw_connection(painter, node, node.right, node.true_label)
    
    def _draw_connection(self, painter, parent, child, condition):
        
        # Aplica transformações de zoom e pan
        start_x = (parent.x + self.pan_x) * self.zoom_factor
        start_y = (parent.y + self.pan_y) * self.zoom_factor
        end_x = (child.x + self.pan_x) * self.zoom_factor
        end_y = (child.y + self.pan_y) * self.zoom_factor
        
        # Cor baseada na condição
        if condition in {"≤", "não"}:
            painter.setPen(QPen(QColor(51, 153, 204), max(1, 2 * self.zoom_factor)))
        else:
            painter.setPen(QPen(QColor(204, 102, 51), max(1, 2 * self.zoom_factor)))
        
        # Desenha linha
        painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
        
        # Só desenha label se o zoom for suficiente para legibilidade
        if self.zoom_factor > 0.5:
            self._draw_edge_label(painter, parent, child, condition, start_x, start_y, end_x, end_y)
    
    def _draw_edge_label(self, painter, parent, child, condition, start_x, start_y, end_x, end_y):
            
        mid_x = (start_x + end_x) / 2
        mid_y = (start_y + end_y) / 2
        
        if parent.condition_text:
            condition_text = condition
        else:
            feature_name = self._feature_name_at(parent.feature)[:12]
            threshold_text = f"{parent.threshold:.3g}" if parent.threshold is not None else "?"
            condition_text = f"{feature_name} {condition} {threshold_text}"

        angle = math.atan2(end_y - start_y, end_x - start_x)

        offset_distance = 20 * self.zoom_factor
        
        perp_x = -math.sin(angle) * offset_distance
        perp_y = math.cos(angle) * offset_distance
        
        label_x = mid_x + perp_x
        label_y = mid_y + perp_y
        
        font = QFont()
        font.setPointSize(max(8, int(10 * self.zoom_factor)))
        painter.setFont(font)
        
        metrics = QFontMetrics(font)
        text_rect = metrics.boundingRect(condition_text)
        
        # Fundo com borda
        bg_rect = QRect(
            int(label_x - text_rect.width()/2 - 4),
            int(label_y - text_rect.height()/2 - 2),
            text_rect.width() + 8,
            text_rect.height() + 4
        )
        
        # Fundo branco com transparência
        painter.fillRect(bg_rect, QColor(255, 255, 255, 240))
        
        # Borda sutil
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRect(bg_rect)
        
        # Texto
        painter.setPen(QPen(QColor(0, 0, 0)))
        painter.drawText(bg_rect, Qt.AlignmentFlag.AlignCenter, condition_text)
    
    def _draw_nodes(self, painter):
        
        for node in self.tree_nodes.values():
            self._draw_node(painter, node)
    
    def _draw_node(self, painter, node):
        
        # Aplica transformações
        x = (node.x + self.pan_x) * self.zoom_factor
        y = (node.y + self.pan_y) * self.zoom_factor
        
        radius = self.node_radius * self.zoom_factor
        
        if node.is_leaf:
            class_colors = [
                QColor(51, 204, 51),   # Verde
                QColor(204, 51, 51),   # Vermelho
                QColor(51, 51, 204),   # Azul
                QColor(204, 204, 51),  # Amarelo
                QColor(204, 51, 204),  # Magenta
            ]
            class_idx = hash(str(node.class_name)) % len(class_colors) if node.class_name else 0
            painter.setBrush(QBrush(class_colors[class_idx]))
        else:
            uncertainty_color = 1.0 - node.uncertainty
            color = QColor(int(uncertainty_color * 255), int(uncertainty_color * 255), 204)
            painter.setBrush(QBrush(color))
        
        painter.setPen(QPen(QColor(0, 0, 0), max(1, int(2 * self.zoom_factor))))
        painter.drawEllipse(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
        
        node_text = self._get_node_text(node)
        font = QFont()
        font.setPointSize(max(7, int(9 * self.zoom_factor)))
        painter.setFont(font)
        
        painter.setPen(QPen(QColor(0, 0, 0)))
        text_rect = QRect(int(x - radius), int(y - radius), int(radius * 2), int(radius * 2))
        painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, node_text)
    
    def _get_node_text(self, node):
        
        if node.is_leaf:
            class_name = str(node.class_name)[:6] if node.class_name else "?"
            return f"{class_name}\n({node.samples})"
        else:
            if node.condition_text:
                label = node.condition_text
                if len(label) > 22:
                    label = label[:19] + "..."
                return f"{label}\n({node.samples})"
            feature_name = self._feature_name_at(node.feature)[:10]
            return f"{feature_name}\n({node.samples})"
    
    def mousePressEvent(self, event):
        
        if event.button() == Qt.MouseButton.LeftButton:
            tree_x = (event.position().x() - self.pan_x) / self.zoom_factor
            tree_y = (event.position().y() - self.pan_y) / self.zoom_factor
            
            clicked_node = self._find_node_at_position(tree_x, tree_y)
            if clicked_node:
                self.node_clicked.emit(clicked_node)
                self._show_node_tooltip(clicked_node, event.position())
                return
            else:
                self.dragging = True
                self.last_pan_x = event.position().x()
                self.last_pan_y = event.position().y()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        super().mousePressEvent(event)
    
    def mouseMoveEvent(self, event):

        if self.dragging:
            dx = event.position().x() - self.last_pan_x
            dy = event.position().y() - self.last_pan_y
            
            self.pan_x += dx
            self.pan_y += dy
            
            self.last_pan_x = event.position().x()
            self.last_pan_y = event.position().y()
            
            self.update()
        
        super().mouseMoveEvent(event)
    
    def mouseReleaseEvent(self, event):
        
        if event.button() == Qt.MouseButton.LeftButton:
            self.dragging = False
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        
        super().mouseReleaseEvent(event)
    
    def _find_node_at_position(self, x, y):
        
        for node in self.tree_nodes.values():
            distance = math.sqrt((x - node.x)**2 + (y - node.y)**2)
            if distance <= self.node_radius:
                return node
        return None
    
    def _show_node_tooltip(self, node, pos):
        
        # Cria conteúdo do tooltip
        info_text = f"Nodo {node.node_id}\n\n"
        
        if node.is_leaf:
            info_text += f"Hoja\nClase: {node.class_name}\nMuestras: {node.samples}"
            if node.values is not None and len(node.values) > 0:
                info_text += f"\nDistribución: {node.values}"
        else:
            if node.condition_text:
                info_text += f"Nó de decisão TREPAN\nTeste: {node.condition_text}\nAmostras reais: {node.samples}"
            else:
                feature_name = self._feature_name_at(node.feature)
                info_text += f"Nó de decisão\nCaracterística: {feature_name}\nLimiar: {node.threshold:.2f}\nAmostras: {node.samples}"
            if self.show_uncertainty:
                info_text += f"\nIncerteza: {node.uncertainty:.2f}"
        
        QMessageBox.information(self, "Detalhes do nó", info_text)
    
    def zoom_in(self):
        
        self.zoom_factor = min(self.zoom_factor * 1.2, 3.0)
        self.center_tree()
    
    def zoom_out(self):
        
        self.zoom_factor = max(self.zoom_factor / 1.2, 0.3)
        self.center_tree()
    
    def reset_zoom(self):
        
        # Recalcula bounds para garantir precisão
        self._calculate_tree_bounds()
        self._auto_fit_tree()
    
    def center_tree(self):
        
        if not self.tree_nodes or self.tree_width == 0 or self.tree_height == 0:
            return
            
        center_x = (self.tree_min_x + self.tree_max_x) / 2
        center_y = (self.tree_min_y + self.tree_max_y) / 2
        
        self.pan_x = (self.width() / 2) / self.zoom_factor - center_x
        self.pan_y = (self.height() / 2) / self.zoom_factor - center_y
        
        self.update()
    
    def toggle_uncertainty_display(self):

        self.show_uncertainty = not self.show_uncertainty
        self.update()
    
    def set_complexity_filter(self, min_samples, max_depth):
        
        self.min_samples_threshold = min_samples
        self.max_depth_display = max_depth
        self._filter_tree()
        self.update()
    
    def _filter_tree(self):
        
        # Remove nós com poucas amostras ou muito profundos
        nodes_to_hide = []
        for node in self.tree_nodes.values():
            if (node.samples < self.min_samples_threshold or 
                self._get_node_depth(node) > self.max_depth_display):
                nodes_to_hide.append(node)
        
        pass
    
    def _get_node_depth(self, node):
        
        depth = 0
        current = node
        while current.node_id != 0:  
            for parent in self.tree_nodes.values():
                if (parent.left == current or parent.right == current):
                    current = parent
                    depth += 1
                    break
            else:
                break
        return depth
    
    def wheelEvent(self, event):
        
        # Obtém o fator de zoom baseado na direção do scroll
        zoom_factor = 1.15 if event.angleDelta().y() > 0 else 1/1.15
        
        old_zoom = self.zoom_factor
        new_zoom = self.zoom_factor * zoom_factor
        
        new_zoom = max(0.1, min(5.0, new_zoom))
        
        if new_zoom != old_zoom:
            mouse_x = event.position().x()
            mouse_y = event.position().y()
            
            tree_x = (mouse_x - self.pan_x) / old_zoom
            tree_y = (mouse_y - self.pan_y) / old_zoom
            
            self.zoom_factor = new_zoom
            
            self.pan_x = mouse_x - tree_x * new_zoom
            self.pan_y = mouse_y - tree_y * new_zoom
            
            self.update()
    
    def resizeEvent(self, event):
        
        super().resizeEvent(event)
        self._auto_fit_tree()
