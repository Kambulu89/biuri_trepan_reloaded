import os
import re
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QSlider, QCheckBox, QDialog, QDialogButtonBox, QFormLayout,
    QSpinBox, QGroupBox, QFrame, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QFont, QPalette


def _sanitize_filename_part(name):
    return re.sub(r'[<>:"/\\|?*\s]', '_', str(name)).strip('_') or "unknown"


def build_tree_export_basename(dataset_name, model_name):
    dataset_stem = os.path.splitext(os.path.basename(dataset_name))[0]
    return f"{_sanitize_filename_part(dataset_stem)}{_sanitize_filename_part(model_name)}"


def export_tree_png(tree_model, feature_names, class_names, output_base, fmt="png", title=None, write_json=False):
    """Exporta a árvore COMPLETA com o mesmo renderer do ecrã (nós elípticos; m-of-n preservado).

    Formatos: png (alta resolução), svg e pdf (vectoriais). Não usa caixas rectangulares nem
    converte m-of-n para CART. Devolve o caminho do ficheiro."""
    from gui.tree_viz.model import build_visualization_model
    from gui.tree_viz.render import export_tree
    base = str(output_base)
    for ext in (".png", ".svg", ".pdf", ".json"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    names = list(feature_names) if feature_names else []
    model = build_visualization_model(tree_model, names, list(class_names or []))
    info = export_tree(model, f"{base}.{fmt}", fmt=fmt, title=title, write_json=write_json)
    return os.path.abspath(info["path"])


# Alias preservado para plugins/testes antigos.
def export_sklearn_tree_png(tree_model, feature_names, class_names, output_base):
    return export_tree_png(tree_model, feature_names, class_names, output_base)


class TreeControlsWidget(QWidget):
    
    
    def __init__(self, tree_widget, dataset_name="dataset", model_name_getter=None, parent=None):
        super().__init__(parent)
        self.tree_widget = tree_widget
        self.dataset_name = dataset_name
        self.model_name_getter = model_name_getter or (lambda: "Arbol")
        self.setup_ui()
        
    def setup_ui(self):
        from gui.tree_viz.strings import tr
        w = self.tree_widget
        layout = QVBoxLayout()
        layout.setSpacing(8)

        title = QLabel("🎛️ Controles del Árbol")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        self.selected_tree_label = QLabel(f"{tr('selected_tree')}: {self.model_name_getter()}")
        self.selected_tree_label.setStyleSheet("font-weight: bold;")
        self.selected_tree_label.setWordWrap(True)
        layout.addWidget(self.selected_tree_label)

        zoom_group = QGroupBox("Zoom y Navegación")
        zl = QVBoxLayout()
        row = QHBoxLayout()
        for text, tip, slot in (("🔍-", "Disminuir zoom", w.zoom_out), ("🔍+", "Aumentar zoom", w.zoom_in)):
            b = QPushButton(text); b.setToolTip(tip); b.clicked.connect(slot); row.addWidget(b)
        zl.addLayout(row)
        row2 = QHBoxLayout()
        for text, tip, slot in (
            ("🔄 Ajustar", "Ajustar la vista a toda el árbol (fit to view)", w.reset_zoom),
            ("↺ Reset", "Restaurar zoom, desplazamiento y selección (sin reconstruir el árbol)", w.reset_view),
        ):
            b = QPushButton(text); b.setToolTip(tip); b.clicked.connect(slot); row2.addWidget(b)
        zl.addLayout(row2)
        row3 = QHBoxLayout()
        for text, tip, slot in (
            ("🎯 Raíz", "Centrar la raíz", w.center_root),
            ("📍 Nodo", "Centrar el nodo seleccionado", w.center_selected),
        ):
            b = QPushButton(text); b.setToolTip(tip); b.clicked.connect(slot); row3.addWidget(b)
        zl.addLayout(row3)
        zoom_group.setLayout(zl)
        layout.addWidget(zoom_group)

        filters_group = QGroupBox("Filtros de Visualización")
        fl = QVBoxLayout()
        self.uncertainty_check = QCheckBox("Mostrar Incertidumbre")
        self.uncertainty_check.setToolTip(tr("uncertainty_def"))
        self.uncertainty_check.setChecked(w.show_uncertainty)
        self.uncertainty_check.toggled.connect(lambda on: w.show_uncertainty != on and w.toggle_uncertainty_display())
        fl.addWidget(self.uncertainty_check)
        self.scientific_check = QCheckBox("Vista científica (IDs, muestras)")
        self.scientific_check.toggled.connect(lambda on: (w.set_scientific_view(on), w.set_show_node_ids(on)))
        fl.addWidget(self.scientific_check)
        advanced_btn = QPushButton("⚙️ Filtros Avanzados")
        advanced_btn.setToolTip("Solo visual: oculta/colapsa nodos; no poda el árbol")
        advanced_btn.clicked.connect(self._show_complexity_dialog)
        fl.addWidget(advanced_btn)
        filters_group.setLayout(fl)
        layout.addWidget(filters_group)

        nav_group = QGroupBox("Navegación")
        nl = QVBoxLayout()
        row4 = QHBoxLayout()
        hp = QPushButton("🛤️ Resaltar Camino"); hp.setCheckable(True); hp.setChecked(True)
        hp.setToolTip("Resalta el camino raíz → nodo seleccionado"); hp.toggled.connect(lambda _: w.toggle_highlight_path())
        col = QPushButton("➕/➖ Colapsar"); col.setToolTip("Colapsar/expandir el subárbol seleccionado (solo visual; doble clic también)")
        col.clicked.connect(lambda: w.toggle_collapse())
        ex = QPushButton("Expandir todo"); ex.clicked.connect(w.expand_all)
        row4.addWidget(hp); row4.addWidget(col); row4.addWidget(ex)
        nl.addLayout(row4)
        from PyQt6.QtWidgets import QLineEdit
        self.search_edit = QLineEdit(); self.search_edit.setPlaceholderText(tr("search"))
        self.search_edit.returnPressed.connect(self._search)
        nl.addWidget(self.search_edit)
        self.search_result = QLabel(""); self.search_result.setWordWrap(True)
        nl.addWidget(self.search_result)
        export_btn = QPushButton("💾 Exportar")
        export_btn.setToolTip("Exportar el árbol completo: PNG (alta resolución), SVG, PDF y JSON")
        export_btn.clicked.connect(self._export_tree)
        nl.addWidget(export_btn)
        nav_group.setLayout(nl)
        layout.addWidget(nav_group)
        layout.addStretch()
        self.setLayout(layout)
        self._search_hits = []
        self._search_idx = 0

    def set_model_name(self, name):
        from gui.tree_viz.strings import tr
        self.selected_tree_label.setText(f"{tr('selected_tree')}: {name}")

    def _search(self):
        from gui.tree_viz.strings import tr
        text = self.search_edit.text()
        if text != getattr(self, "_search_text", None):
            self._search_text = text
            self._search_hits = self.tree_widget.search(text)
            self._search_idx = 0
        else:
            self._search_idx += 1
        if not self._search_hits:
            self.search_result.setText(tr("no_results"))
            return
        self._search_idx %= len(self._search_hits)
        nid = self._search_hits[self._search_idx]
        self.search_result.setText(f"{self._search_idx + 1}/{len(self._search_hits)}: " + ", ".join(f"N{i}" for i in self._search_hits[:12]))
        self.tree_widget.select_node(nid, center=True)

    def _show_complexity_dialog(self):
        """Filtros SÓ VISUAIS: esconde/colapsa (indicando '+N nós'); nunca poda a árvore."""
        w = self.tree_widget
        max_depth = w.model.depth if w.model is not None else 10
        max_samples = max([n.samples or 0 for n in w.model.nodes.values()] + [1]) if w.model is not None else 100
        dialog = QDialog(self)
        dialog.setWindowTitle("Filtros de Complejidad (solo visuales)")
        dialog.setModal(True)
        dialog.resize(420, 260)
        layout = QVBoxLayout()
        note = QLabel("Estos filtros solo cambian la visualización. No podan ni modifican el árbol.")
        note.setWordWrap(True)
        layout.addWidget(note)
        form = QFormLayout()
        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setRange(0, max_samples)
        self.samples_slider.setValue(int(w.min_samples_threshold or 0))
        self.samples_label = QLabel(str(self.samples_slider.value()))
        self.samples_slider.valueChanged.connect(self._on_samples_change)
        sl = QHBoxLayout(); sl.addWidget(self.samples_slider); sl.addWidget(self.samples_label)
        form.addRow("Colapsar nodos con menos muestras que (0 = sin filtro):", sl)
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setRange(0, max_depth)
        self.depth_slider.setValue(max_depth if w.max_depth_display is None else int(w.max_depth_display))
        self.depth_label = QLabel(str(self.depth_slider.value()))
        self.depth_slider.valueChanged.connect(self._on_depth_change)
        dl = QHBoxLayout(); dl.addWidget(self.depth_slider); dl.addWidget(self.depth_label)
        form.addRow("Mostrar hasta profundidad:", dl)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
                                   | QDialogButtonBox.StandardButton.Reset)
        buttons.accepted.connect(lambda: self._apply_filters(dialog, max_depth))
        buttons.rejected.connect(dialog.reject)
        buttons.button(QDialogButtonBox.StandardButton.Reset).clicked.connect(lambda: (w.clear_filters(), dialog.reject()))
        layout.addWidget(buttons)
        dialog.setLayout(layout)
        dialog.exec()

    def _on_samples_change(self, value):
        self.samples_label.setText(str(value))

    def _on_depth_change(self, value):
        self.depth_label.setText(str(value))

    def _apply_filters(self, dialog, tree_depth=None):
        min_samples = self.samples_slider.value()
        depth = self.depth_slider.value()
        self.tree_widget.set_complexity_filter(min_samples or None, None if (tree_depth is not None and depth >= tree_depth) else depth)
        dialog.accept()

    def _export_tree(self):
        tree_model = self.tree_widget.tree_model
        if tree_model is None or self.tree_widget.model is None:
            QMessageBox.warning(self, "Exportar Árbol", "Ningún árbol disponible para exportar.")
            return
        default_basename = build_tree_export_basename(self.dataset_name, self.model_name_getter())
        file_path, selected = QFileDialog.getSaveFileName(
            self, "Exportar Árbol", f"{default_basename}.png",
            "PNG alta resolución (*.png);;SVG vectorial (*.svg);;PDF vectorial (*.pdf)",
        )
        if not file_path:
            return
        fmt = "svg" if "svg" in selected.lower() else "pdf" if "pdf" in selected.lower() else "png"
        lowered = file_path.lower()
        for ext in ("png", "svg", "pdf"):
            if lowered.endswith("." + ext):
                fmt = ext
        try:
            info = self.tree_widget.export(file_path, fmt=fmt, title=str(self.model_name_getter()), write_json=True)
            QMessageBox.information(self, "Exportar Árbol",
                                    f"Árbol exportado con éxito!\n\n📁 {info['path']}\n🧾 {info.get('json', '')}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar árbol:\n{str(e)}")
