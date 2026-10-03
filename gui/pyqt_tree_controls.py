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


def export_tree_png(tree_model, feature_names, class_names, output_base, fmt='png'):
    """Exporta sklearn ou TREPAN histórico sem converter m-of-n para CART.

    ``fmt`` aceita ``png`` (omissão), ``svg`` ou ``pdf``; só muda o formato da imagem.
    """
    import matplotlib.pyplot as plt

    fmt = (fmt or 'png').lower().lstrip('.')
    if fmt not in ('png', 'svg', 'pdf'):
        raise ValueError(f"Formato de imagem não suportado: {fmt}")
    output_path = output_base if output_base.lower().endswith(f'.{fmt}') else f'{output_base}.{fmt}'
    feature_names_list = list(feature_names) if feature_names else None

    if hasattr(tree_model, 'root_') and hasattr(tree_model, 'export_text'):
        try:
            from core.trepan_original import TrepanOriginalExtractor
            import graphviz
            adapter = TrepanOriginalExtractor()
            adapter.adopt_tree(tree_model)
            dot_path = f'{output_base}.dot'
            adapter.export_tree_image(feature_names_list, class_names, dot_path, open_image=False)
            dot_data = Path(dot_path).read_text(encoding='utf-8')
            rendered = graphviz.Source(dot_data).render(
                filename=output_base, format=fmt, cleanup=True
            )
            return os.path.abspath(rendered)
        except (ImportError, OSError, RuntimeError):
            # Fallback sem Graphviz: não altera a árvore; apresenta as regras nativas.
            fig, ax = plt.subplots(figsize=(16, 10))
            ax.axis('off')
            ax.text(
                0.01, 0.99, tree_model.export_text(class_names),
                va='top', ha='left', family='monospace', fontsize=8, wrap=True,
            )
            fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close(fig)
            return os.path.abspath(output_path)

    from sklearn.tree import plot_tree
    names = [str(c) for c in class_names] if class_names is not None else None
    depth = tree_model.get_depth() if hasattr(tree_model, 'get_depth') else 5
    n_leaves = tree_model.get_n_leaves() if hasattr(tree_model, 'get_n_leaves') else 10
    fig, ax = plt.subplots(figsize=(max(12, n_leaves * 1.5), max(8, depth * 2)))
    try:
        plot_tree(
            tree_model, feature_names=feature_names_list, class_names=names,
            filled=True, rounded=True, ax=ax, fontsize=9,
        )
        fig.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    finally:
        plt.close(fig)
    return os.path.abspath(output_path)


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
        
        layout = QVBoxLayout()
        layout.setSpacing(10)
        
        # Título
        title = QLabel("🎛️ Controles del Árbol")
        title.setFont(QFont("Arial", 16, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Controles de zoom
        zoom_group = QGroupBox("Zoom y Navegación")
        zoom_layout = QHBoxLayout()
        
        zoom_out_btn = QPushButton("🔍-")
        zoom_out_btn.setToolTip("Disminuir zoom")
        zoom_out_btn.clicked.connect(self.tree_widget.zoom_out)
        
        zoom_label = QLabel("Zoom")
        zoom_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        zoom_in_btn = QPushButton("🔍+")
        zoom_in_btn.setToolTip("Aumentar zoom")
        zoom_in_btn.clicked.connect(self.tree_widget.zoom_in)
        
        reset_zoom_btn = QPushButton("🔄 Ajustar a la Pantalla")
        reset_zoom_btn.setToolTip("Ajustar zoom para mostrar todo el árbol")
        reset_zoom_btn.clicked.connect(self.tree_widget.reset_zoom)
        
        center_btn = QPushButton("🎯 Centrar")
        center_btn.setToolTip("Centrar el árbol")
        center_btn.clicked.connect(self.tree_widget.center_tree)
        
        zoom_layout.addWidget(zoom_out_btn)
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(zoom_in_btn)
        zoom_layout.addWidget(reset_zoom_btn)
        zoom_layout.addWidget(center_btn)
        
        zoom_group.setLayout(zoom_layout)
        layout.addWidget(zoom_group)
        
        # Controles de filtros
        filters_group = QGroupBox("Filtros de Visualización")
        filters_layout = QVBoxLayout()
        
        # Checkbox para incerteza
        uncertainty_layout = QHBoxLayout()
        self.uncertainty_check = QCheckBox("Mostrar Incertidumbre")
        self.uncertainty_check.setChecked(True)
        self.uncertainty_check.toggled.connect(self.tree_widget.toggle_uncertainty_display)
        uncertainty_layout.addWidget(self.uncertainty_check)
        uncertainty_layout.addStretch()
        filters_layout.addLayout(uncertainty_layout)
        
        # Botão para filtros avançados
        advanced_btn = QPushButton("⚙️ Filtros Avanzados")
        advanced_btn.clicked.connect(self._show_complexity_dialog)
        filters_layout.addWidget(advanced_btn)
        
        filters_group.setLayout(filters_layout)
        layout.addWidget(filters_group)
        
        # Controles de navegação
        navigation_group = QGroupBox("Navegación")
        navigation_layout = QHBoxLayout()
        
        highlight_path_btn = QPushButton("🛤️ Resaltar Camino")
        highlight_path_btn.setToolTip("Resaltar caminos importantes")
        highlight_path_btn.clicked.connect(self._highlight_important_paths)
        
        export_btn = QPushButton("💾 Exportar")
        export_btn.setToolTip("Exportar árbol actual")
        export_btn.clicked.connect(self._export_tree)
        
        navigation_layout.addWidget(highlight_path_btn)
        navigation_layout.addWidget(export_btn)
        
        navigation_group.setLayout(navigation_layout)
        layout.addWidget(navigation_group)
        
        # Espaçador
        layout.addStretch()
        
        self.setLayout(layout)
        
    def _show_complexity_dialog(self):
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Filtros de Complejidad")
        dialog.setModal(True)
        dialog.setMinimumSize(320, 240); dialog.resize(400, 300)
        
        layout = QVBoxLayout()
        
        # Título
        title = QLabel("Filtros de Complejidad")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Formulário
        form_layout = QFormLayout()
        
        # Slider para mínimo de amostras
        self.samples_slider = QSlider(Qt.Orientation.Horizontal)
        self.samples_slider.setMinimum(1)
        self.samples_slider.setMaximum(100)
        self.samples_slider.setValue(10)
        self.samples_slider.valueChanged.connect(self._on_samples_change)
        
        self.samples_label = QLabel("10")
        self.samples_label.setMinimumWidth(30)
        
        samples_layout = QHBoxLayout()
        samples_layout.addWidget(self.samples_slider)
        samples_layout.addWidget(self.samples_label)
        
        form_layout.addRow("Mín. Muestras:", samples_layout)
        
        # Slider para profundidade máxima
        self.depth_slider = QSlider(Qt.Orientation.Horizontal)
        self.depth_slider.setMinimum(1)
        self.depth_slider.setMaximum(10)
        self.depth_slider.setValue(5)
        self.depth_slider.valueChanged.connect(self._on_depth_change)
        
        self.depth_label = QLabel("5")
        self.depth_label.setMinimumWidth(30)
        
        depth_layout = QHBoxLayout()
        depth_layout.addWidget(self.depth_slider)
        depth_layout.addWidget(self.depth_label)
        
        form_layout.addRow("Profundidad Máx:", depth_layout)
        
        layout.addLayout(form_layout)
        
        # Botões
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(lambda: self._apply_filters(dialog))
        buttons.rejected.connect(dialog.reject)
        
        layout.addWidget(buttons)
        
        dialog.setLayout(layout)
        dialog.exec()
    
    def _on_samples_change(self, value):
        
        self.samples_label.setText(str(value))
    
    def _on_depth_change(self, value):
        
        self.depth_label.setText(str(value))
    
    def _apply_filters(self, dialog):
        
        min_samples = self.samples_slider.value()
        max_depth = self.depth_slider.value()
        
        self.tree_widget.set_complexity_filter(min_samples, max_depth)
        dialog.accept()
    
    def _highlight_important_paths(self):
        
        # Implementação simplificada - em uma versão completa,
        # isso destacaria caminhos com alta importância ou baixa incerteza
        from PyQt6.QtWidgets import QMessageBox
        QMessageBox.information(self, "Resaltar Caminos", 
                               "La funcionalidad de resaltar caminos se implementará próximamente.")
    
    def _export_tree(self):
        tree_model = self.tree_widget.tree_model
        if tree_model is None:
            QMessageBox.warning(self, "Exportar Árbol", "Ningún árbol disponible para exportar.")
            return

        default_basename = build_tree_export_basename(
            self.dataset_name, self.model_name_getter()
        )
        default_path = f"{default_basename}.png"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar Árbol",
            default_path,
            "Imágenes PNG (*.png);;Todos los archivos (*.*)",
        )
        if not file_path:
            return

        output_base = file_path
        if output_base.lower().endswith(".png"):
            output_base = output_base[:-4]

        try:
            saved_path = export_tree_png(
                tree_model,
                self.tree_widget.feature_names,
                self.tree_widget.class_names,
                output_base,
            )
            QMessageBox.information(
                self,
                "Exportar Árbol",
                f"Árbol exportado con éxito!\n\n📁 {saved_path}",
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Error al exportar árbol:\n{str(e)}",
            )
