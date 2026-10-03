import sys
import os
import warnings
import re
import difflib
from pathlib import Path

warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*sipPyTypeDict.*")

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QTextEdit, QFrame, QScrollArea, QFileDialog,
    QMessageBox, QProgressBar, QSplitter, QGroupBox, QGridLayout,
    QTabWidget, QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
    QDialog, QDialogButtonBox, QFormLayout, QLineEdit, QComboBox, QSpinBox,
    QCheckBox, QSlider, QProgressDialog, QDoubleSpinBox, QSizePolicy,
    QToolButton, QGraphicsOpacityEffect,
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import (
    QFont, QPixmap, QPainter, QLinearGradient, QColor, QIcon, QPalette,
    QShortcut, QKeySequence,
)
from PyQt6.QtCore import QRect, QSize
import numpy as np
import pandas as pd
from scipy.io import arff

sys.path.append(str(Path(__file__).parent.parent))
from core.trepan import TrepanReloaded
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.ontology_processor import OntologyProcessor
from core.protocol_audit import stable_sha256_strings
from core.ontology_quality import OntologyQualityGate
from core.ontology_reasoner import run_owl_reasoner
from core.arff_schema import parse_arff_class_order
from core.metrics_comparator import MetricsComparator
from core.mlp_trainer import MLPTrainer
from core.natural_language_explainer import NaturalLanguageExplainer

from gui.pyqt_tree_widget import InteractiveTreeWidget, TreeDetailsPanel
from gui.pyqt_tree_controls import TreeControlsWidget
from gui.pyqt_explanation_widget import ExplanationWidget
from gui.pyqt_metrics_visualizer import MetricsVisualizer
from gui.training_worker import TrainingWorker
from gui.counterfactual_worker import CounterfactualWorker
from gui.counterfactual_panel import CounterfactualPanel
from gui.surrogate_improvement_dialog import SurrogateImprovementDialog
from gui.metrics_worker import MetricsWorker
from gui import theme
from gui.audit_panel import AuditPanel
from gui.audit_controller import AuditController, dataset_fingerprint_of, progress_text
from gui.strings import tr
from core.training_config import (
    get_training_preset, TRAINING_PRESETS, enforce_scientific_preset,
    resolve_trepan_structure_limits, PRODUCTION_TRAINING_PRESET,
)
from core.performance_logger import PerformanceLogger
from core.model_cache import (
    build_cache_key,
    compute_dataset_hash,
    compute_ontology_hash,
    compute_preprocessing_hash,
    load_cached_training,
    save_cached_training,
    serialize_mlp_trainer_state,
    restore_mlp_trainer_state,
)


class GradientFrame(QFrame):
    
    def __init__(self, color1, color2, direction='horizontal'):
        super().__init__()
        self.color1 = QColor(color1)
        self.color2 = QColor(color2)
        self.direction = direction
        
    def paintEvent(self, event):
        painter = QPainter(self)
        gradient = QLinearGradient(0, 0, 
                                   self.width() if self.direction == 'horizontal' else 0,
                                   self.height() if self.direction == 'vertical' else 0)
        gradient.setColorAt(0, self.color1)
        gradient.setColorAt(1, self.color2)
        painter.fillRect(self.rect(), gradient)


class ClarityTrail(QWidget):
    """Opaque → clear path: Datos → Modelo → Árbol.

    Operate-mode delight: milestone feedback tied to BIURI's mechanism,
    not a celebration on every click.
    """

    STEPS = (
        ("datos", "Dados"),
        ("modelo", "Modelo"),
        ("arbol", "Árvore"),
    )

    def __init__(self, parent=None):
        super().__init__(parent)
        self._done = {key: False for key, _ in self.STEPS}
        self._chips = {}
        self._pulse_anims = []
        self._build()

    def _build(self):
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        for index, (key, label) in enumerate(self.STEPS):
            if index:
                connector = QLabel("→")
                connector.setStyleSheet(
                    "color: rgba(255,255,255,0.55); font-size: 12px; font-weight: bold;"
                )
                row.addWidget(connector)

            chip = QLabel(f"{index + 1} {label}")
            chip.setAlignment(Qt.AlignmentFlag.AlignCenter)
            chip.setMinimumWidth(72)
            chip.setAccessibleName(f"Etapa {label}")
            self._chips[key] = chip
            self._apply_chip_style(key)
            row.addWidget(chip)

    def _apply_chip_style(self, key):
        chip = self._chips[key]
        if self._done[key]:
            chip.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba(255, 255, 255, 0.95);
                    color: {theme.BRAND_DARK};
                    border-radius: 12px;
                    padding: 4px 10px;
                    font-size: 11px;
                    font-weight: bold;
                }}
            """)
        else:
            chip.setStyleSheet(f"""
                QLabel {{
                    background-color: rgba(255, 255, 255, 0.12);
                    color: {theme.TEXT_ON_BRAND_MUTED};
                    border: 1px solid rgba(255, 255, 255, 0.35);
                    border-radius: 12px;
                    padding: 4px 10px;
                    font-size: 11px;
                    font-weight: bold;
                }}
            """)

    def mark(self, step, done=True, animate=True):
        if step not in self._done:
            return
        was_done = self._done[step]
        self._done[step] = bool(done)
        self._apply_chip_style(step)
        if done and not was_done and animate and not theme.prefers_reduced_motion():
            self._pulse(step)

    def reset(self):
        for key in self._done:
            self._done[key] = False
            self._apply_chip_style(key)

    def _pulse(self, step):
        chip = self._chips.get(step)
        if chip is None:
            return
        effect = chip.graphicsEffect()
        if not isinstance(effect, QGraphicsOpacityEffect):
            effect = QGraphicsOpacityEffect(chip)
            chip.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(420)
        anim.setStartValue(0.35)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        self._pulse_anims.append(anim)
        anim.finished.connect(
            lambda a=anim: self._pulse_anims.remove(a) if a in self._pulse_anims else None
        )
        anim.start()


class ActionButton(QPushButton):
    
    def __init__(self, text, icon_text, color, parent=None, compact=False):
        super().__init__(text, parent)
        self.icon_text = icon_text
        self.color = color
        self.compact = compact
        self.setup_ui()
        
    def setup_ui(self):
        self.setMinimumHeight(36 if self.compact else 48)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        font_size = 12 if self.compact else 14
        padding = 8 if self.compact else 10
        hover = theme.lighten_action(self.color)
        pressed = theme.darken_action(self.color)
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.color};
                color: {theme.TEXT_ON_BRAND};
                border: none;
                border-radius: {theme.RADIUS}px;
                font-size: {font_size}px;
                font-weight: bold;
                padding: {padding}px;
                text-align: left;
            }}
            QPushButton:hover {{
                background-color: {hover};
            }}
            QPushButton:pressed {{
                background-color: {pressed};
            }}
            QPushButton:disabled {{
                background-color: {theme.DISABLED_FILL};
                color: {theme.TEXT_DISABLED};
            }}
        """)


class DataLoadDialog(QDialog):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cargar Datos y Ontología")
        self.setModal(True)
        self.setMinimumSize(480, 420)
        self.resize(600, 550)
        
        layout = QVBoxLayout()
        
        # Título
        title_label = QLabel("Cargar Datos ARFF y Ontología OWL")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(title_label)
        
        # Seleção de arquivo ARFF
        arff_group = QGroupBox("📁 Archivo de Datos (ARFF)")
        arff_layout = QHBoxLayout()
        self.arff_path_edit = QLineEdit()
        self.arff_path_edit.setPlaceholderText("Seleccione un archivo ARFF...")
        self.arff_path_edit.setAccessibleName("Ruta del archivo ARFF")
        arff_browse_btn = QPushButton("Buscar")
        arff_browse_btn.setAccessibleName("Buscar archivo ARFF")
        arff_browse_btn.clicked.connect(lambda: self.browse_file("arff"))

        arff_layout.addWidget(self.arff_path_edit)
        arff_layout.addWidget(arff_browse_btn)
        arff_group.setLayout(arff_layout)

        # Seleção de arquivo OWL
        owl_group = QGroupBox("📚 Archivo de Ontología (OWL) - Opcional")
        owl_layout = QHBoxLayout()
        self.owl_path_edit = QLineEdit()
        self.owl_path_edit.setPlaceholderText("Seleccione un archivo OWL (opcional)...")
        self.owl_path_edit.setAccessibleName("Ruta del archivo OWL opcional")
        owl_browse_btn = QPushButton("Buscar")
        owl_browse_btn.setAccessibleName("Buscar archivo OWL")
        owl_browse_btn.clicked.connect(lambda: self.browse_file("owl"))

        owl_layout.addWidget(self.owl_path_edit)
        owl_layout.addWidget(owl_browse_btn)
        owl_group.setLayout(owl_layout)

        # Opções de carregamento
        options_group = QGroupBox("Opciones de Carga")
        options_layout = QFormLayout()

        self.encoding_combo = QComboBox()
        self.encoding_combo.addItems(["utf-8", "latin-1", "cp1252"])
        self.encoding_combo.setAccessibleName("Codificación del archivo")

        options_layout.addRow("Codificación:", self.encoding_combo)

        options_group.setLayout(options_layout)

        info_label = QLabel(
            "<b>Consejo:</b><br>"
            "• El archivo ARFF es <b>obligatorio</b><br>"
            "• El archivo OWL es <b>opcional</b>: si lo proporciona, se usará "
            "para enriquecer los datos con conocimiento semántico<br>"
            "• Sin OWL, los datos se cargan sin enriquecimiento ontológico"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            f"background-color: #e3f2fd; color: {theme.TEXT}; "
            "padding: 10px; border-radius: 5px; margin-top: 10px;"
        )

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout.addWidget(arff_group)
        layout.addWidget(owl_group)
        layout.addWidget(options_group)
        layout.addWidget(info_label)
        layout.addWidget(buttons)

        self.setLayout(layout)
        QWidget.setTabOrder(self.arff_path_edit, arff_browse_btn)
        QWidget.setTabOrder(arff_browse_btn, self.owl_path_edit)
        QWidget.setTabOrder(self.owl_path_edit, owl_browse_btn)
        QWidget.setTabOrder(owl_browse_btn, self.encoding_combo)
        
    def browse_file(self, file_type):
        if file_type == "arff":
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Seleccione archivo de datos ARFF", 
                "", 
                "Archivos ARFF (*.arff)"
            )
            if file_path:
                self.arff_path_edit.setText(file_path)
        elif file_type == "owl":
            file_path, _ = QFileDialog.getOpenFileName(
                self, 
                "Seleccione archivo de ontología OWL", 
                "", 
                "Archivos OWL (*.owl);;Archivos RDF (*.rdf);;Todos los archivos (*)"
            )
            if file_path:
                self.owl_path_edit.setText(file_path)


class OntologyLoadDialog(QDialog):
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Cargar Ontología")
        self.setModal(True)
        self.setMinimumSize(400, 240)
        self.resize(500, 300)
        
        layout = QVBoxLayout()
        
        file_layout = QHBoxLayout()
        self.file_path_edit = QLineEdit()
        self.file_path_edit.setPlaceholderText("Seleccione un archivo OWL...")
        self.file_path_edit.setAccessibleName("Ruta del archivo OWL")
        browse_btn = QPushButton("Buscar")
        browse_btn.setAccessibleName("Buscar archivo OWL")
        browse_btn.clicked.connect(self.browse_file)

        file_layout.addWidget(self.file_path_edit)
        file_layout.addWidget(browse_btn)

        info_label = QLabel(
            "<b>Formato de ontología:</b><br>"
            "• Archivos OWL (.owl)<br>"
            "• Ontologías RDF/XML<br>"
            "• Ontologías OWL/XML<br><br>"
            "<b>Uso:</b><br>"
            "Cargue una ontología de dominio para usar Trepan-Reloaded "
            "con conocimiento semántico enriquecido."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet(
            f"background-color: #f0f0f0; color: {theme.TEXT}; "
            "padding: 10px; border-radius: 5px;"
        )
        
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        
        layout.addLayout(file_layout)
        layout.addWidget(info_label)
        layout.addWidget(buttons)
        
        self.setLayout(layout)
        
    def browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "Seleccione archivo de ontología", 
            "", 
            "Archivos OWL (*.owl);;Archivos RDF (*.rdf);;Todos los archivos (*)"
        )
        if file_path:
            self.file_path_edit.setText(file_path)


class TreeVisualizationWidget(QWidget):

    def __init__(self, tree_model=None, feature_names=None, class_names=None, 
                 trepan_original_tree=None, trepan_reloaded_tree=None, c45_tree=None,
                 trepan_original_improved_tree=None,
                 trepan_reloaded_improved_tree=None,
                 dataset_name="dataset", feature_names_reloaded=None,
                 primary_tree_label="Árvore principal"):
        super().__init__()
        self.tree_model = tree_model  # Árvore padrão (para compatibilidade)
        self.feature_names = feature_names or []
        self.feature_names_reloaded = feature_names_reloaded or self.feature_names
        self.class_names = class_names or []
        self.dataset_name = dataset_name
        self.primary_tree_label = primary_tree_label
        
        # Armazenar todas as árvores disponíveis
        self.trepan_original_tree = trepan_original_tree
        self.trepan_reloaded_tree = trepan_reloaded_tree
        self.c45_tree = c45_tree
        self.trepan_original_improved_tree = trepan_original_improved_tree
        self.trepan_reloaded_improved_tree = trepan_reloaded_improved_tree
        
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        # Tab already labels this surface — skip a second title band
        if any(tree is not None for tree in (
            self.tree_model,
            self.trepan_original_tree,
            self.trepan_reloaded_tree,
            self.c45_tree,
            self.trepan_original_improved_tree,
            self.trepan_reloaded_improved_tree,
        )):
            self.create_interactive_tree_visualization(layout)
        else:
            placeholder = QLabel(
                "Ningún árbol disponible.\n"
                "Treine um modelo e use «Visualizar árvore» nas ferramentas avançadas."
            )
            placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
            placeholder.setWordWrap(True)
            placeholder.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 14px; padding: 24px;")
            layout.addWidget(placeholder)

        self.setLayout(layout)
    
    def _feature_names_for_tree(self, tree):
        """Escolhe nomes pelo n_features da árvore (Original=30, Reloaded=91/212, etc.)."""
        if tree is None:
            return self.feature_names
        n_feat = int(getattr(getattr(tree, 'tree_', None), 'n_features', 0) or 0)

        candidates = []
        seen = set()
        for names in (
            self.feature_names_reloaded,
            getattr(self, '_extractor_tree_names', None),
            self.feature_names,
        ):
            if not names:
                continue
            key = tuple(names)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(list(names))

        if n_feat:
            exact = [c for c in candidates if len(c) == n_feat]
            if exact:
                return exact[0]
            wider = [c for c in candidates if len(c) > n_feat]
            if wider:
                return wider[0][:n_feat]

        is_reloaded = (
            tree is self.trepan_reloaded_tree
            or tree is self.trepan_reloaded_improved_tree
        )
        if is_reloaded and self.feature_names_reloaded:
            names = list(self.feature_names_reloaded)
        else:
            names = list(self.feature_names)

        if n_feat and len(names) < n_feat:
            names = names + [f"feature_{i}" for i in range(len(names), n_feat)]
        elif n_feat and len(names) > n_feat:
            names = names[:n_feat]
        return names

    def create_interactive_tree_visualization(self, layout):
        
        try:
            # Determinar qual árvore usar por padrão
            default_tree = None
            tree_options = []
            
            if self.trepan_original_tree:
                tree_options.append(("Trepan-Original", self.trepan_original_tree))
                if not default_tree:
                    default_tree = self.trepan_original_tree
            
            if self.trepan_reloaded_tree:
                tree_options.append(("Trepan-Reloaded", self.trepan_reloaded_tree))
                if not default_tree:
                    default_tree = self.trepan_reloaded_tree

            if self.trepan_original_improved_tree is not None:
                tree_options.append((
                    "Trepan-Original — melhorada por CF",
                    self.trepan_original_improved_tree,
                ))

            if self.trepan_reloaded_improved_tree is not None:
                tree_options.append((
                    "Trepan-Reloaded — melhorada por CF",
                    self.trepan_reloaded_improved_tree,
                ))
            
            if self.c45_tree:
                tree_options.append(("C4.5-Nativo", self.c45_tree))
                if not default_tree:
                    default_tree = self.c45_tree

            # A árvore CF é entregue como ``tree_model`` e não como uma das
            # variantes Trepan/C4.5. O código anterior entrava neste método,
            # mas ignorava essa árvore e acabava por apresentar o placeholder
            # "Ningún árbol disponible" apesar de o treino ter terminado.
            # Usamos a árvore principal apenas como fallback para preservar a
            # selecção já existente quando há árvores Trepan/C4.5 disponíveis.
            if default_tree is None and self.tree_model is not None:
                tree_options.append((self.primary_tree_label, self.tree_model))
                default_tree = self.tree_model
            
            if not default_tree:
                placeholder = QLabel("Ningún árbol disponible para visualización")
                placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
                placeholder.setStyleSheet(f"color: {theme.TEXT_SECONDARY}; font-size: 14px;")
                layout.addWidget(placeholder)
                return

            # Também conservar a opção única: os controlos de exportação usam
            # este nome para identificar correctamente a árvore apresentada.
            self.tree_options = tree_options
            
            # Criar seletor de árvore se temos múltiplas opções
            if len(tree_options) > 1:
                tree_selection_layout = QHBoxLayout()
                tree_selection_label = QLabel("Seleccione el árbol:")
                tree_selection_label.setStyleSheet("font-weight: bold;")
                
                self.tree_combo = QComboBox()
                for tree_name, _ in tree_options:
                    self.tree_combo.addItem(tree_name)
                
                tree_selection_layout.addWidget(tree_selection_label)
                tree_selection_layout.addWidget(self.tree_combo)
                tree_selection_layout.addStretch()
                
                layout.addLayout(tree_selection_layout)
                
                # Conectar o combo box usando a referência armazenada
                self.tree_combo.currentIndexChanged.connect(
                    lambda idx: self.switch_tree(self.tree_options[idx][1])
                )
            
            # Criar widget de árvore interativa com melhorias
            default_names = self._feature_names_for_tree(default_tree)
            self.tree_widget = InteractiveTreeWidget(
                default_tree,
                default_names,
                self.class_names,
                algorithm=tree_options[0][0] if tree_options else None,
            )
            # Painel de detalhes sob demanda (nó/aresta seleccionados)
            self.tree_details = TreeDetailsPanel(self.tree_widget)
            
            # Criar controles da árvore
            self.tree_controls = self._create_tree_controls()
            
            # Layout com controles e árvore
            self.main_layout = QHBoxLayout()
            
            # Controles à esquerda
            controls_frame = QFrame()
            controls_frame.setMinimumWidth(200)
            controls_frame.setMaximumWidth(320)
            controls_frame.setSizePolicy(
                QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
            )
            controls_frame.setFrameStyle(QFrame.Shape.Box)
            controls_frame.setStyleSheet(
                f"background-color: {theme.SURFACE_CONTROL}; "
                f"border: 1px solid {theme.BORDER_CONTROL};"
            )

            controls_layout = QVBoxLayout()
            controls_layout.addWidget(self.tree_controls)
            controls_frame.setLayout(controls_layout)

            self.main_layout.addWidget(controls_frame)

            # Árvore ao centro + painel de detalhes à direita
            self.main_layout.addWidget(self.tree_widget, 4)
            self.main_layout.addWidget(self.tree_details, 1)

            layout.addLayout(self.main_layout)

        except Exception as e:
            error_label = QLabel(f"Error al crear visualización interactiva: {str(e)}")
            error_label.setStyleSheet(f"color: {theme.ACTION_METRICS};")
            layout.addWidget(error_label)
    
    def _get_current_model_name(self):
        if hasattr(self, 'tree_combo'):
            return self.tree_combo.currentText()
        if hasattr(self, 'tree_options') and self.tree_options:
            return self.tree_options[0][0]
        return "Arbol"

    def _create_tree_controls(self):
        return TreeControlsWidget(
            self.tree_widget,
            dataset_name=self.dataset_name,
            model_name_getter=self._get_current_model_name,
        )

    def switch_tree(self, new_tree):
        
        if hasattr(self, 'main_layout') and new_tree and hasattr(self, 'tree_widget'):
            try:
                names = self._feature_names_for_tree(new_tree)
                # Só selecciona a árvore já construída e volta a desenhar (sem retreino).
                self.tree_widget.update_tree(
                    new_tree, feature_names=names,
                    algorithm=self.tree_combo.currentText() if hasattr(self, 'tree_combo') else None,
                )
                
                # Recriar controles (export PNG usa feature_names actuais)
                self.tree_controls.close()
                self.tree_controls.deleteLater()
                self.tree_controls = self._create_tree_controls()
                
                # Limpar layout anterior
                for i in reversed(range(self.main_layout.count())):
                    item = self.main_layout.itemAt(i)
                    if item:
                        widget = item.widget()
                        if widget:
                            widget.setParent(None)
                
                # Recriar controles frame
                controls_frame = QFrame()
                controls_frame.setMinimumWidth(200)
                controls_frame.setMaximumWidth(320)
                controls_frame.setSizePolicy(
                    QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding
                )
                controls_frame.setFrameStyle(QFrame.Shape.Box)
                controls_frame.setStyleSheet(
                    f"background-color: {theme.SURFACE_CONTROL}; "
                    f"border: 1px solid {theme.BORDER_CONTROL};"
                )

                controls_layout = QVBoxLayout()
                controls_layout.addWidget(self.tree_controls)
                controls_frame.setLayout(controls_layout)

                # Adicionar de volta
                self.main_layout.addWidget(controls_frame)
                self.main_layout.addWidget(self.tree_widget, 4)
                self.main_layout.addWidget(self.tree_details, 1)

            except Exception as e:
                print(f"Error ao alternar árvore: {e}")
                import traceback
                traceback.print_exc()


class MetricsComparisonWidget(QWidget):
    
    def __init__(self):
        super().__init__()
        self.setup_ui()
        
    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.metrics_visualizer = MetricsVisualizer()
        layout.addWidget(self.metrics_visualizer)

        self.setLayout(layout)
    
    def update_data_info(self, data, meta):
        
        import pandas as pd
        import numpy as np
        
        try:
            # Converter dados para DataFrame se necessário
            if isinstance(data, tuple):
                X, y = data
                # Criar DataFrame com features
                if hasattr(meta, 'features') or 'features' in meta:
                    feature_names = meta['features'] if 'features' in meta else [f"feature_{i}" for i in range(X.shape[1])]
                else:
                    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                
                df = pd.DataFrame(X, columns=feature_names)
                
                # Adicionar coluna target se tiver meta target
                if 'target' in meta:
                    df[meta['target']] = y
                else:
                    df['target'] = y
            elif isinstance(data, pd.DataFrame):
                df = data.copy()
            else:
                df = pd.DataFrame(data)
            
            # Passar dados para o visualizador
            self.metrics_visualizer.set_data_info(df, meta)
        except Exception as e:
            print(f"Error ao atualizar dados do visualizador: {e}")

    def update_comparison_results(self, comparison_results):
        """Atualiza o visualizador com os resultados reais da última comparação."""
        if not comparison_results:
            return

        self.metrics_visualizer.clear_model_data()
        self.metrics_visualizer.set_comparison_results(comparison_results)

        precision = comparison_results.get('precision', {})
        fidelity = comparison_results.get('fidelity', {})
        n_samples = comparison_results.get('model_info', {}).get('n_test_samples')

        # O MLP aparece como referência preditiva no gráfico de Precisão Macro.
        # A fidelidade é exibida somente para os dois modelos TREPAN, pois é
        # nesses substitutos que a concordância com o oráculo é comparável.
        model_entries = [
            ('MLP Original', 'mlp', None),
            ('MLP Ontológico', 'mlp_ontological', None),
            ('Trepan-Original', 'trepan_original', 'trepan_original'),
            ('C4.5-Nativo', 'c45_j48', 'c45_j48'),
            ('Trepan-Reloaded', 'trepan_reloaded', 'trepan_reloaded'),
        ]

        for display_name, prec_key, fid_key in model_entries:
            prec_data = precision.get(prec_key)
            if prec_data is None:
                continue

            prec_val = (prec_data.get('precision_macro') if prec_data.get('precision_macro') is not None else prec_data.get('precision') or 0) * 100
            accuracy_val = (prec_data.get('accuracy') or 0) * 100
            # Fidelity só existe com Oracle (TREPAN). O C4.5 mostra concordância com o MLP (diagnóstico);
            # MLPs não têm fidelity: nunca se apresenta 0.0% como se fosse um valor medido.
            fid_val = 0.0
            fid_raw = (fidelity.get(fid_key) or {}).get('overall_fidelity') if fid_key else None
            if fid_raw is not None:
                fid_val = fid_raw * 100
            fid_kind = None if fid_raw is None else ('agreement' if fid_key == 'c45_j48' else 'oracle')
            fid_block = (fidelity.get(fid_key) or {}) if fid_key else {}

            print(
                f"[DIAG] GUI -> {display_name}: "
                f"Precisão Macro={prec_val:.1f}%, Exatidão(auditoria)={accuracy_val:.1f}%, "
                f"Fidelity={'n/a (sem Oracle)' if fid_kind is None else f'{fid_val:.1f}%'}"
            )
            self.metrics_visualizer.add_model_data(
                display_name,
                prec_val,
                fid_val,
                accuracy=accuracy_val,
                sample_size=n_samples,
                balanced_accuracy=(prec_data.get('balanced_accuracy') or 0) * 100,
                macro_f1=(prec_data.get('f1_macro') or 0) * 100,
                fidelity_kind=fid_kind,
                oracle=(fidelity.get(fid_key) or {}).get('fidelity_reference') if fid_key else 'rótulo real',
                feature_space=(fidelity.get(fid_key) or {}).get('feature_space', 'original') if fid_key else 'original',
                canonical=True,
                fidelity_to_active_oracle=(
                    (fid_block.get('fidelity_to_active_oracle') or fid_block.get('active_oracle_fidelity')) * 100
                    if fid_block.get('fidelity_to_active_oracle') is not None or fid_block.get('active_oracle_fidelity') is not None
                    else None
                ),
                fidelity_to_mlp_original=(
                    fid_block.get('fidelity_to_mlp_original') * 100
                    if fid_block.get('fidelity_to_mlp_original') is not None else None
                ),
            )


class BiuriApp(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.trepan = TrepanReloaded(use_default_ontology=False)
        self.metrics_comparator = MetricsComparator()
        self.natural_explainer = None  # Será inicializado quando necessário
        
        self.current_data = None
        self.original_data = None
        self.augmented_data = None
        self.mlp_model = None
        self.mlp_model_reloaded = None
        self.mlp_model_onto = None
        self.mlp_model_residual = None
        self.selected_oracle = None
        self.selected_oracle_label = None
        self.ontology_acceptance = None
        self.mlp_trainer_reloaded = MLPTrainer()
        self.explanation_generated = False
        self.loaded_ontology = None
        self.loaded_ontology_path = None
        self.loaded_ontology_metadata = {}
        self.trepan_original_tree = None
        self.trepan_reloaded_tree = None
        self.c45_tree = None
        self.trepan_original_audit = None
        self.trepan_reloaded_audit = None
        self.X_encoded = None
        self.y_encoded = None
        self.X_encoded_aug = None
        self.y_encoded_aug = None
        self._eval_split_original = None
        self._eval_split_augmented = None
        self.feature_names_original = None
        self.feature_names_augmented = None
        self.trepan_reloaded_feature_names = None
        self.ontology_augmented_columns = []
        self.ontology_feature_summary = []
        self._ontology_match_cache = {}
        self._ontology_matcher = None
        self.ontology_match_threshold = 0.72
        self.ontology_quality_gate = OntologyQualityGate()
        self.ontology_quality_report = None
        self.ontology_reasoner_report = None
        self.last_ontology_load_diagnostic = None
        self.ontology_transformer = None
        self.onto_feature_bias_weight = TrepanReloadedExtractor.ONTO_FEATURE_BIAS_WEIGHT_DEFAULT
        self._training_worker = None
        self._training_cancel_flag = False
        self._training_progress_dialog = None
        self._cf_worker = None
        self._cf_progress_dialog = None
        self.cf_result = None
        self.cf_interactive_result = None
        self.cf_global_result = None
        self.cf_tree_result = None
        self.cf_transfer_result = None
        self.cf_improve_result = None
        self.trepan_improved_tree = None
        self.trepan_reloaded_improved_tree = None
        self._metrics_worker = None
        self._metrics_progress_dialog = None
        self._counterfactual_modules_loaded = False
        # Camada de auditoria (observabilidade): só lê o estado; nunca recalcula resultados.
        self.current_seed = 42  # seed fixa usada pelo pipeline da GUI (random_state=42)
        self.dataset_info = {}
        self.dataset_fingerprint = None
        self._last_cache_info = None
        self._last_training_preset = None
        self._pending_failure_detail = None
        self.audit = None

        self.setup_ui()
        self.setup_connections()
        self.audit = AuditController(self, self.audit_panel)
        self.audit.apply_buttons()
        self._set_status(
            "BIURI traduce redes opacas en árboles legibles. Empiece cargando un ARFF."
        )

    def setup_ui(self):

        self.setWindowTitle("BIURI - IA Explicable")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(960, 640)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout()
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        self.create_header(main_layout)

        self.create_main_content(main_layout)

        central_widget.setLayout(main_layout)
        self._setup_status_bar()

    def _setup_status_bar(self):
        bar = self.statusBar()
        bar.setSizeGripEnabled(True)
        # Colors come from theme.APP_QSS

    def _set_status(self, message, timeout_ms=0):
        self.statusBar().showMessage(message, timeout_ms)

    def _unlock_clarity(self, step, status=None, nudge_advanced=False):
        if hasattr(self, "clarity_trail"):
            self.clarity_trail.mark(step, True, animate=True)
        if status:
            self._set_status(status)
        if nudge_advanced:
            self._nudge_advanced_tools()

    def _nudge_advanced_tools(self):
        """One soft cue that the tree tools are ready — never auto-opens the panel."""
        if not hasattr(self, "advanced_toggle"):
            return
        if theme.prefers_reduced_motion():
            self._set_status(
                "Herramientas avanzadas listas (Ctrl+Shift+A).", timeout_ms=4000
            )
            return
        toggle = self.advanced_toggle
        base = toggle.styleSheet()
        toggle.setStyleSheet(f"""
            QToolButton {{
                font-size: 13px;
                font-weight: bold;
                color: {theme.BRAND};
                padding: 8px 4px;
                border: none;
                text-align: left;
            }}
        """)
        QTimer.singleShot(1400, lambda: toggle.setStyleSheet(base))

    def _sidebar_group_style(self, title_size=14):
        return f"""
            QGroupBox {{
                font-size: {title_size}px;
                font-weight: bold;
                color: {theme.TEXT};
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS}px;
                margin-top: {theme.SPACE_3}px;
                padding-top: {theme.SPACE_2}px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px 0 4px;
            }}
        """

    def create_header(self, parent_layout):
        # Compact brand bar — Operate mode: reclaim height for the work surface
        header_frame = GradientFrame(theme.BRAND, theme.ACCENT, 'horizontal')
        header_frame.setFixedHeight(56)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(
            theme.SPACE_4, theme.SPACE_2, theme.SPACE_4, theme.SPACE_2
        )
        header_layout.setSpacing(theme.SPACE_3)

        logo_label = QLabel("🧠")
        logo_label.setStyleSheet("font-size: 22px;")

        text_col = QVBoxLayout()
        text_col.setSpacing(0)
        text_col.setContentsMargins(0, 0, 0, 0)

        title_label = QLabel("BIURI - IA Explicable")
        title_label.setStyleSheet(f"""
            font-size: 18px;
            font-weight: bold;
            color: {theme.TEXT_ON_BRAND};
        """)

        subtitle_label = QLabel(
            "Transforme modelos complexos em explicações claras e compreensíveis"
        )
        subtitle_label.setWordWrap(True)
        subtitle_label.setStyleSheet(f"""
            font-size: 11px;
            color: {theme.TEXT_ON_BRAND_MUTED};
        """)

        text_col.addWidget(title_label)
        text_col.addWidget(subtitle_label)

        header_layout.addWidget(logo_label, 0, Qt.AlignmentFlag.AlignVCenter)
        header_layout.addLayout(text_col, 1)

        self.clarity_trail = ClarityTrail()
        self.clarity_trail.setToolTip(
            "Percurso de clareza: dados → modelo opaco → árvore legível"
        )
        header_layout.addWidget(
            self.clarity_trail, 0, Qt.AlignmentFlag.AlignVCenter
        )

        header_frame.setLayout(header_layout)
        parent_layout.addWidget(header_frame)

    def create_main_content(self, parent_layout):
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(6)
        splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {theme.BORDER};
            }}
            QSplitter::handle:hover {{
                background-color: {theme.BRAND};
            }}
        """)

        splitter.addWidget(self.create_sidebar())
        splitter.addWidget(self.create_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([280, 1100])

        content_wrapper = QWidget()
        content_layout = QHBoxLayout(content_wrapper)
        content_layout.setContentsMargins(12, 12, 12, 12)
        content_layout.setSpacing(0)
        content_layout.addWidget(splitter)

        parent_layout.addWidget(content_wrapper, 1)

    def create_sidebar(self):
        sidebar_root = QWidget()
        sidebar_root.setMinimumWidth(260)
        sidebar_root.setMaximumWidth(380)
        root_layout = QVBoxLayout(sidebar_root)
        # Spacing scale: 4 / 8 / 12 / 16 — tight within groups, 12 between regions
        root_layout.setSpacing(12)
        root_layout.setContentsMargins(0, 0, 4, 0)

        # Primary path pinned above scroll so it never leaves the viewport
        pipeline_group = QGroupBox("1 · Fluxo de trabalho")
        pipeline_group.setStyleSheet(self._sidebar_group_style(title_size=14))
        pipeline_layout = QVBoxLayout()
        pipeline_layout.setSpacing(8)
        pipeline_layout.setContentsMargins(8, 12, 8, 8)

        self.btn_load_data = ActionButton("📁 Carregar dados", "📁", theme.ACTION_LOAD)
        self.btn_train_model = ActionButton("🤖 Treinar modelo", "🤖", theme.ACTION_TRAIN)
        self.btn_generate_explanation = ActionButton(
            "💡 Gerar explicação", "💡", theme.ACTION_EXPLAIN
        )

        pipeline_layout.addWidget(self.btn_load_data)

        train_cluster = QWidget()
        train_cluster_layout = QVBoxLayout(train_cluster)
        train_cluster_layout.setContentsMargins(0, 0, 0, 0)
        train_cluster_layout.setSpacing(4)

        self.training_mode_combo = QComboBox()
        self.training_mode_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        # Produção V9.2: treino bloqueado no modo científico.
        scientific = TRAINING_PRESETS[PRODUCTION_TRAINING_PRESET]
        self.training_mode_combo.addItem(scientific.display_name, scientific.key)
        self.training_mode_combo.setItemData(
            0, scientific.description, Qt.ItemDataRole.ToolTipRole
        )
        self.training_mode_combo.setCurrentIndex(0)
        self.training_mode_combo.setEnabled(False)
        self.training_mode_combo.setToolTip(
            "Modo Científico obrigatório na V9.2 de produção."
        )

        self.training_cache_checkbox = QCheckBox("Usar caché de modelos")
        self.training_cache_checkbox.setChecked(True)
        self.training_cache_checkbox.setStyleSheet(
            "font-size: 12px; font-weight: normal;"
        )

        train_cluster_layout.addWidget(self.btn_train_model)
        train_cluster_layout.addWidget(self.training_mode_combo)
        train_cluster_layout.addWidget(self.training_cache_checkbox)
        pipeline_layout.addWidget(train_cluster)
        pipeline_layout.addWidget(self.btn_generate_explanation)

        self.btn_load_data.setAccessibleName("Cargar datos ARFF")
        self.btn_load_data.setAccessibleDescription(
            "Abre el diálogo para cargar un archivo ARFF y una ontología OWL opcional"
        )
        self.btn_train_model.setAccessibleName("Entrenar modelo")
        self.btn_train_model.setAccessibleDescription(
            "Entrena el MLP y extrae árboles Trepan / Trepan-Reloaded"
        )
        self.btn_generate_explanation.setAccessibleName("Generar explicación")
        self.btn_generate_explanation.setAccessibleDescription(
            "Genera reglas legibles a partir del modelo entrenado"
        )
        self.training_mode_combo.setAccessibleName("Modo de entrenamiento")
        self.training_cache_checkbox.setAccessibleName("Usar caché de modelos entrenados")
        pipeline_group.setLayout(pipeline_layout)
        root_layout.addWidget(pipeline_group)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet(f"""
            QScrollArea {{
                background-color: transparent;
                border: none;
            }}
            QScrollBar:vertical {{
                width: 8px;
                background: {theme.SURFACE};
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical {{
                background: {theme.BORDER};
                border-radius: 4px;
                min-height: 30px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {theme.BRAND};
            }}
        """)

        secondary = QWidget()
        secondary_layout = QVBoxLayout(secondary)
        secondary_layout.setSpacing(12)
        secondary_layout.setContentsMargins(2, 0, 2, 0)

        self.ontology_settings_group = QGroupBox("Ontología Trepan-Reloaded")
        self.ontology_settings_group.setStyleSheet(self._sidebar_group_style(title_size=13))
        self.ontology_settings_group.setVisible(False)
        ontology_settings_layout = QVBoxLayout()
        ontology_settings_layout.setSpacing(8)
        ontology_settings_layout.setContentsMargins(8, 12, 8, 8)

        onto_bias_header = QHBoxLayout()
        onto_bias_label = QLabel("Peso de sesgo onto_*")
        onto_bias_label.setWordWrap(True)
        onto_bias_label.setStyleSheet(
            f"font-size: 12px; font-weight: normal; color: {theme.TEXT_SECONDARY};"
        )
        self.onto_bias_value_label = QLabel("2.5")
        self.onto_bias_value_label.setStyleSheet(
            f"font-size: 13px; font-weight: bold; color: {theme.BRAND}; min-width: 28px;"
        )
        onto_bias_header.addWidget(onto_bias_label)
        onto_bias_header.addStretch()
        onto_bias_header.addWidget(self.onto_bias_value_label)

        self.onto_bias_slider = QSlider(Qt.Orientation.Horizontal)
        self.onto_bias_slider.setRange(10, 50)
        self.onto_bias_slider.setValue(25)
        self.onto_bias_slider.setTickInterval(5)
        self.onto_bias_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.onto_bias_slider.setEnabled(False)

        self.onto_bias_spin = QDoubleSpinBox()
        self.onto_bias_spin.setRange(1.0, 5.0)
        self.onto_bias_spin.setSingleStep(0.1)
        self.onto_bias_spin.setDecimals(1)
        self.onto_bias_spin.setValue(self.onto_feature_bias_weight)
        self.onto_bias_spin.setEnabled(False)
        self.onto_bias_spin.setSuffix("×")

        onto_bias_hint = QLabel(
            "1.0–5.0. Valores mayores priorizan divisiones en features onto_*."
        )
        onto_bias_hint.setWordWrap(True)
        onto_bias_hint.setStyleSheet(
            f"font-size: 11px; color: {theme.TEXT_SECONDARY}; font-weight: normal;"
        )

        ontology_settings_layout.addLayout(onto_bias_header)
        ontology_settings_layout.addWidget(self.onto_bias_slider)
        ontology_settings_layout.addWidget(self.onto_bias_spin)
        ontology_settings_layout.addWidget(onto_bias_hint)
        self.ontology_settings_group.setLayout(ontology_settings_layout)
        secondary_layout.addWidget(self.ontology_settings_group)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Ferramentas avançadas")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setChecked(False)
        self.advanced_toggle.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.advanced_toggle.setStyleSheet(f"""
            QToolButton {{
                font-size: 13px;
                font-weight: bold;
                color: {theme.TEXT_SECONDARY};
                padding: 8px 4px;
                border: none;
                text-align: left;
            }}
            QToolButton:hover {{
                color: {theme.BRAND};
            }}
        """)
        self.advanced_toggle.toggled.connect(self._on_advanced_toggled)
        secondary_layout.addWidget(self.advanced_toggle)

        self.advanced_container = QWidget()
        self.advanced_container.setVisible(False)
        advanced_layout = QVBoxLayout(self.advanced_container)
        advanced_layout.setSpacing(6)
        advanced_layout.setContentsMargins(0, 0, 0, 0)

        self.btn_visualize_tree = ActionButton(
            "🌳 Visualizar árvore", "🌳", theme.ACTION_LOAD, compact=True
        )
        self.btn_compare_metrics = ActionButton(
            "📊 Comparar métricas", "📊", theme.ACTION_METRICS, compact=True
        )
        self.btn_natural_explanations = ActionButton(
            "🧠 Explicações naturais", "🧠", theme.ACTION_NL, compact=True
        )
        self.btn_generate_counterfactuals = ActionButton(
            "🔀 Gerar contrafactuais", "🔀", theme.ACTION_CF, compact=True
        )
        self.btn_improve_surrogate = ActionButton(
            "📈 Melhorar árvore substituta", "📈", theme.ACTION_IMPROVE, compact=True
        )
        self.btn_export_results = ActionButton(
            "💾 " + tr("action.export_results"), "💾", theme.ACTION_EXPORT, compact=True
        )
        self.btn_export_tree = ActionButton(
            "🖼 " + tr("action.export_tree"), "🖼", theme.ACTION_EXPORT, compact=True
        )

        for btn in (
            self.btn_visualize_tree,
            self.btn_compare_metrics,
            self.btn_natural_explanations,
            self.btn_generate_counterfactuals,
            self.btn_improve_surrogate,
            self.btn_export_results,
            self.btn_export_tree,
        ):
            advanced_layout.addWidget(btn)

        self.btn_visualize_tree.setAccessibleName("Visualizar árbol")
        self.btn_compare_metrics.setAccessibleName("Comparar métricas")
        self.btn_natural_explanations.setAccessibleName("Explicaciones en lenguaje natural")
        self.btn_generate_counterfactuals.setAccessibleName("Generar contrafactuales")
        self.btn_improve_surrogate.setAccessibleName("Mejorar árbol sustituto")
        self.btn_export_results.setAccessibleName(tr("action.export_results"))
        self.btn_export_tree.setAccessibleName(tr("action.export_tree"))
        self.advanced_toggle.setAccessibleName("Ferramentas avançadas")
        self.onto_bias_slider.setAccessibleName("Peso de sesgo ontológico")
        self.onto_bias_spin.setAccessibleName("Peso de sesgo ontológico numérico")

        secondary_layout.addWidget(self.advanced_container)
        secondary_layout.addStretch()
        scroll_area.setWidget(secondary)
        root_layout.addWidget(scroll_area, 1)
        return sidebar_root

    def _on_advanced_toggled(self, checked):
        self.advanced_container.setVisible(checked)
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def create_results_panel(self):
        results_frame = GradientFrame(theme.BRAND, theme.ACCENT, 'vertical')
        results_frame.setMinimumHeight(320)
        results_frame.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )

        results_layout = QVBoxLayout()
        results_layout.setContentsMargins(
            theme.SPACE_3, theme.SPACE_3, theme.SPACE_3, theme.SPACE_3
        )
        results_layout.setSpacing(0)

        self.content_tabs = QTabWidget()
        self.content_tabs.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.content_tabs.tabBar().setUsesScrollButtons(True)
        self.content_tabs.setStyleSheet(f"""
            QTabWidget::pane {{
                border: 1px solid rgba(255, 255, 255, 0.3);
                background-color: rgba(255, 255, 255, 0.95);
                border-radius: {theme.RADIUS}px;
            }}
            QTabBar::tab {{
                background-color: rgba(255, 255, 255, 0.3);
                color: {theme.TEXT_ON_BRAND};
                padding: 8px 16px;
                margin-right: 2px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }}
            QTabBar::tab:selected {{
                background-color: {theme.SURFACE_RAISED};
                color: {theme.TEXT};
            }}
        """)

        self.results_tab = QWidget()
        self.setup_results_tab()
        self.content_tabs.addTab(self.results_tab, "📋 Resultados")

        self.visualization_tab = QWidget()
        self.setup_visualization_tab()
        self.content_tabs.addTab(self.visualization_tab, "🌳 Visualização")

        self.metrics_tab = QWidget()
        self.setup_metrics_tab()
        self.content_tabs.addTab(self.metrics_tab, "📊 Métricas")

        self.audit_panel = AuditPanel()
        self.content_tabs.addTab(self.audit_panel, "🔍 " + tr("tab.audit"))

        self.counterfactual_tab = CounterfactualPanel()
        self.counterfactual_tab.configure(
            n_instances=0, class_labels={}, available_models=[], dataset_name="",
        )
        self.content_tabs.addTab(self.counterfactual_tab, "🔀 Contrafactuais")

        results_layout.addWidget(self.content_tabs)
        results_frame.setLayout(results_layout)
        return results_frame

    def setup_results_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(0)

        self.results_text = QTextEdit()
        self.results_text.setStyleSheet(f"""
            QTextEdit {{
                background-color: {theme.SURFACE_RAISED};
                border: none;
                border-radius: {theme.RADIUS}px;
                padding: {theme.SPACE_3}px;
                font-size: 14px;
                color: {theme.TEXT};
            }}
        """)

        initial_text = """
BIURI — de la caja negra al árbol legible

Un MLP decide; usted necesita entender por qué.
BIURI entrena el modelo y extrae un árbol sustituto (Trepan / Trepan-Reloaded)
cuyas reglas puede leer, comparar y mejorar.

Empiece en el panel izquierdo:
1. Cargar datos (ARFF) — ontología OWL opcional
2. Entrenar modelo — el oráculo opaco
3. Generar explicación — reglas en lenguaje claro

Observe la barra superior: Datos → Modelo → Árbol se ilumina a medida
que la red se vuelve interpretable.
        """

        self.results_text.setPlainText(initial_text.strip() + "\n")
        layout.addWidget(self.results_text)

        self.results_tab.setLayout(layout)
        
    def setup_visualization_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.tree_widget = TreeVisualizationWidget()
        layout.addWidget(self.tree_widget)

        self.visualization_tab.setLayout(layout)

    def setup_metrics_tab(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.metrics_widget = MetricsComparisonWidget()
        layout.addWidget(self.metrics_widget)

        self.metrics_tab.setLayout(layout)
        
    def setup_connections(self):
        if hasattr(self, "clarity_trail"):
            self.clarity_trail.setAccessibleName("Percurso de clareza: Dados, Modelo, Árvore")
        if hasattr(self, "content_tabs"):
            self.content_tabs.setAccessibleName("Panel de resultados y explicaciones")
        if hasattr(self, "results_text"):
            self.results_text.setAccessibleName("Registro de resultados")
            self.results_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)

        self.btn_load_data.clicked.connect(self.load_data)
        self.btn_train_model.clicked.connect(self.train_model)
        self.btn_generate_explanation.clicked.connect(self.generate_explanation)
        self.btn_visualize_tree.clicked.connect(self.visualize_tree)
        self.btn_compare_metrics.clicked.connect(self.compare_metrics)
        self.btn_natural_explanations.clicked.connect(self.show_natural_explanations)
        self.btn_export_results.clicked.connect(self.export_results)
        self.btn_export_tree.clicked.connect(self.export_tree)
        self.btn_generate_counterfactuals.clicked.connect(self.generate_counterfactuals_action)
        self.btn_improve_surrogate.clicked.connect(self.improve_surrogate_action)
        self.counterfactual_tab.generate_requested.connect(
            self._generate_cf_from_panel
        )
        self.counterfactual_tab.global_requested.connect(
            self._generate_global_cf_from_panel
        )
        self.counterfactual_tab.tree_requested.connect(
            self._build_cf_tree_from_panel
        )
        self.counterfactual_tab.visualize_tree_requested.connect(
            self._visualize_cf_tree
        )
        self.counterfactual_tab.transfer_requested.connect(
            self._evaluate_cf_transfer_from_panel
        )
        self.counterfactual_tab.export_requested.connect(
            self._export_cf_from_panel
        )
        self.onto_bias_spin.valueChanged.connect(self._on_onto_bias_spin_changed)
        self.onto_bias_slider.valueChanged.connect(self._on_onto_bias_slider_changed)

        self._setup_shortcuts()
        self._setup_tab_order()
        self._apply_shortcut_tooltips()

    def _setup_shortcuts(self):
        """Keyboard affordances for the Operate pipeline (do not steal text-field keys)."""
        bindings = (
            ("Ctrl+O", self.load_data, "Cargar datos"),
            ("Ctrl+T", self.train_model, "Entrenar modelo"),
            ("Ctrl+E", self.generate_explanation, "Generar explicación"),
            ("Ctrl+Shift+V", self.visualize_tree, "Visualizar árbol"),
            ("Ctrl+Shift+M", self.compare_metrics, "Comparar métricas"),
            ("Ctrl+Shift+A", self._toggle_advanced_tools, "Herramientas avanzadas"),
            ("F1", self._show_keyboard_help, "Ayuda de atajos"),
        )
        self._shortcuts = []
        for sequence, slot, _label in bindings:
            shortcut = QShortcut(QKeySequence(sequence), self)
            shortcut.setContext(Qt.ShortcutContext.ApplicationShortcut)
            shortcut.activated.connect(slot)
            self._shortcuts.append(shortcut)

    def _setup_tab_order(self):
        """Sidebar task path first, then results tabs."""
        order = [
            self.btn_load_data,
            self.btn_train_model,
            self.training_mode_combo,
            self.training_cache_checkbox,
            self.btn_generate_explanation,
            self.advanced_toggle,
            self.btn_visualize_tree,
            self.btn_compare_metrics,
            self.btn_natural_explanations,
            self.btn_generate_counterfactuals,
            self.btn_improve_surrogate,
            self.btn_export_results,
            self.content_tabs,
        ]
        for prev, nxt in zip(order, order[1:]):
            QWidget.setTabOrder(prev, nxt)

    def _apply_shortcut_tooltips(self):
        tips = {
            self.btn_load_data: "Cargar ARFF / OWL (Ctrl+O)",
            self.btn_train_model: "Entrenar MLP y árboles (Ctrl+T)",
            self.btn_generate_explanation: "Generar reglas legibles (Ctrl+E)",
            self.btn_visualize_tree: "Abrir Visualización (Ctrl+Shift+V)",
            self.btn_compare_metrics: "Comparar métricas (Ctrl+Shift+M)",
            self.advanced_toggle: "Mostrar u ocultar herramientas (Ctrl+Shift+A)",
        }
        for widget, tip in tips.items():
            existing = widget.toolTip()
            widget.setToolTip(f"{existing} — {tip}" if existing else tip)

    def _toggle_advanced_tools(self):
        if hasattr(self, "advanced_toggle"):
            self.advanced_toggle.toggle()

    def _show_keyboard_help(self):
        QMessageBox.information(
            self,
            "Atajos de teclado — BIURI",
            "Ctrl+O — Cargar datos\n"
            "Ctrl+T — Entrenar modelo\n"
            "Ctrl+E — Generar explicación\n"
            "Ctrl+Shift+V — Visualizar árbol\n"
            "Ctrl+Shift+M — Comparar métricas\n"
            "Ctrl+Shift+A — Herramientas avanzadas\n"
            "F1 — Esta ayuda\n\n"
            "Tab recorre el flujo de trabajo de izquierda a derecha.",
        )

    def _busy_workers_running(self):
        for worker in (
            getattr(self, "_training_worker", None),
            getattr(self, "_metrics_worker", None),
            getattr(self, "_cf_worker", None),
        ):
            if worker is not None and worker.isRunning():
                return True
        return False

    def _warn_if_busy(self, action_label="esta acción"):
        if self._busy_workers_running():
            QMessageBox.warning(
                self,
                "Operación en curso",
                f"Espere a que termine el proceso actual antes de {action_label}.\n"
                "Puede cancelar el diálogo de progreso si está disponible.",
            )
            return True
        return False

    def _set_onto_bias_controls_enabled(self, enabled):
        self.onto_bias_slider.setEnabled(enabled)
        self.onto_bias_spin.setEnabled(enabled)
        if hasattr(self, "ontology_settings_group"):
            self.ontology_settings_group.setVisible(bool(enabled))

    def _clear_loaded_ontology(self):
        """ARFF sem OWL: garante modo compatibilidade (sem ontologia activa)."""
        self.loaded_ontology = None
        self.loaded_ontology_path = None
        self.loaded_ontology_metadata = {}
        self._ontology_match_cache = {}
        self._ontology_matcher = None
        self.ontology_augmented_columns = []
        self.ontology_feature_summary = []
        self.augmented_data = None
        self.feature_names_augmented = None
        self.ontology_quality_report = None
        self.ontology_reasoner_report = None
        self.last_ontology_load_diagnostic = None
        self.ontology_transformer = None
        self.trepan.clear_ontology()
        self._set_onto_bias_controls_enabled(False)

    def _sync_onto_bias_controls_from_value(self, value):
        value = float(value)
        self.onto_bias_spin.blockSignals(True)
        self.onto_bias_slider.blockSignals(True)
        self.onto_bias_spin.setValue(value)
        self.onto_bias_slider.setValue(int(round(value * 10)))
        self.onto_bias_value_label.setText(f"{value:.1f}")
        self.onto_bias_spin.blockSignals(False)
        self.onto_bias_slider.blockSignals(False)

    def _apply_onto_bias_weight(self, value=None):
        if value is None:
            value = self.onto_bias_spin.value()
        self.onto_feature_bias_weight = float(value)
        if hasattr(self.trepan, 'extractor') and self.trepan.extractor is not None:
            self.trepan.extractor.set_onto_feature_bias_weight(value)
        if self._ontology_matcher is not None:
            self._ontology_matcher.set_onto_feature_bias_weight(value)
        self._audit_check_stale()

    def _on_onto_bias_spin_changed(self, value):
        self._sync_onto_bias_controls_from_value(value)
        self._apply_onto_bias_weight(value)

    def _on_onto_bias_slider_changed(self, int_value):
        value = int_value / 10.0
        self._sync_onto_bias_controls_from_value(value)
        self._apply_onto_bias_weight(value)
        
    def load_data(self):
        if self._warn_if_busy("cargar nuevos datos"):
            return

        dialog = DataLoadDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            arff_path = dialog.arff_path_edit.text()
            owl_path = dialog.owl_path_edit.text()
            
            if not arff_path:
                QMessageBox.warning(self, "Advertencia", "¡Por favor, seleccione un archivo ARFF!")
                return
            
            try:
                # Primeiro carregar ontologia se fornecida; ARFF só → modo sem ontologia
                if owl_path:
                    self.show_progress("Cargando ontología...")
                    ontology_loaded = self.load_ontology_from_file(owl_path)
                    if ontology_loaded:
                        self.show_results("✅ ¡Ontología cargada con éxito!\n\nCargando datos...")
                    else:
                        self.show_progress("Cargando datos sin enriquecimiento OWL...")
                else:
                    self._clear_loaded_ontology()

                # Depois carregar dados ARFF
                self.show_progress("Cargando datos...")
                self.load_data_from_file(arff_path)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Error al cargar archivos: {str(e)}")
                    
    def _extract_arff_header_info(self, file_path):
        relation_name = None
        attributes = []
        num_instances = 0
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            for line in lines:
                line_stripped = line.strip()
                
                if not line_stripped or line_stripped.startswith('%'):
                    continue

                if line_stripped.upper().startswith('@RELATION'):
                    relation_name = line_stripped[9:].strip()  # Remove '@RELATION'
                    relation_name = relation_name.strip('"\'')
                
                elif line_stripped.upper().startswith('@ATTRIBUTE'):
                    attr_line = line_stripped[10:].strip()  # Remove '@ATTRIBUTE'
                    
                    attr_name = None
                    attr_type = 'unknown'
                    
                    if attr_line.startswith('"') or attr_line.startswith("'"):
                        quote_char = attr_line[0]
                        end_quote_idx = attr_line.find(quote_char, 1)
                        if end_quote_idx != -1:
                            attr_name = attr_line[1:end_quote_idx]
                            remaining = attr_line[end_quote_idx + 1:].strip()
                        else:
                            parts = attr_line.split(None, 1)
                            attr_name = parts[0].strip('"\'')
                            remaining = parts[1] if len(parts) > 1 else ''
                    else:
                        parts = attr_line.split(None, 1)
                        attr_name = parts[0].strip('"\'')
                        remaining = parts[1] if len(parts) > 1 else ''
                    
                    if remaining:
                        attr_type = remaining.strip()
                    
                    if attr_name:
                        attributes.append({'name': attr_name, 'type': attr_type})

                elif line_stripped.upper().startswith('@DATA'):
                    data_start_idx = lines.index(line)
                    for data_line in lines[data_start_idx + 1:]:
                        stripped_data_line = data_line.strip()
                        if stripped_data_line and not stripped_data_line.startswith('%'):
                            num_instances += 1
                    break
                    
        except Exception as e:
            print(f"Error ao extrair cabeçalho ARFF: {e}")
            
        return relation_name, attributes, num_instances
    
    def _normalize_ontology_token(self, value):
        
        if value is None:
            return ""
        if isinstance(value, bytes):
            try:
                value = value.decode("utf-8")
            except Exception:
                value = value.decode("latin-1", errors="ignore")
        return re.sub(r"[^a-z0-9]+", "", str(value).lower())

    def _get_ontology_matcher(self):
        """Matcher partilhado com TrepanReloadedExtractor (fuzzy logic unificada)."""
        if self.loaded_ontology is None:
            return None
        if (
            self._ontology_matcher is None
            or self._ontology_matcher.ontology is not self.loaded_ontology
        ):
            self._ontology_matcher = TrepanReloadedExtractor(
                ontology=self.loaded_ontology,
                onto_feature_bias_weight=self.onto_feature_bias_weight,
            )
        return self._ontology_matcher

    def _get_ontology_matching_entities(self):
        """Entidades OWL para matching de features (Class + Datatype/ObjectProperty)."""
        if self.loaded_ontology is None:
            return [], {}
        matcher = self._get_ontology_matcher()
        if matcher is None:
            return [], {}
        return matcher._get_ontology_matching_entities(self.loaded_ontology)

    def _find_best_ontology_match(self, label, ontology_entities=None, *, for_class_values=False):
        if not label:
            return None, 0.0, None

        matcher = self._get_ontology_matcher()
        if matcher is None:
            return None, 0.0, None

        if for_class_values:
            entities = matcher._get_ontology_class_entities_for_value_matching(
                self.loaded_ontology
            )
        elif ontology_entities is None:
            pairs, _ = self._get_ontology_matching_entities()
            entities = TrepanReloadedExtractor._unwrap_matching_entities(pairs)
        else:
            entities = TrepanReloadedExtractor._unwrap_matching_entities(ontology_entities)

        if not entities:
            return None, 0.0, None

        cache_key = f"{'classval' if for_class_values else 'feat'}:{label}"
        if cache_key in self._ontology_match_cache:
            return self._ontology_match_cache[cache_key]

        match_result = matcher._find_matching_concept(
            str(label),
            entities,
            allow_target_classes=for_class_values,
        )
        if match_result and match_result.get('score', 0) >= self.ontology_match_threshold:
            concept_name = match_result.get('matched_entity') or match_result.get('concept')
            entity_type = match_result.get('entity_type')
            concept_obj = next(
                (c for c in entities if getattr(c, 'name', None) == concept_name),
                None,
            )
            result = (concept_obj, float(match_result['score']), entity_type)
        else:
            result = (
                None,
                float(match_result['score']) if match_result else 0.0,
                match_result.get('entity_type') if match_result else None,
            )

        self._ontology_match_cache[cache_key] = result
        return result

    def _map_series_values_to_ontology(self, series):
        
        mapped_values = []
        mapped_scores = []
        for value in series.astype(str):
            concept_obj, score, _entity_type = self._find_best_ontology_match(
                value, for_class_values=True
            )
            if concept_obj is not None:
                mapped_values.append(getattr(concept_obj, "name", str(value)))
                mapped_scores.append(round(score, 3))
            else:
                mapped_values.append(value)
                mapped_scores.append(0.0)
        return mapped_values, mapped_scores

    def _collect_entity_enrichment_metadata(self, entity_obj, entity_name, matcher):
        """Metadados para colunas onto_* (Class, DatatypeProperty ou ObjectProperty)."""
        parents = []
        depth = 0
        num_children = 0
        num_properties = 0
        entity_type = matcher._get_ontology_entity_type(entity_obj)

        if entity_type == 'class':
            try:
                parents = [
                    getattr(parent, "name", str(parent))
                    for parent in getattr(entity_obj, "is_a", [])
                    if getattr(parent, "name", None)
                ]
            except Exception:
                pass
            try:
                ancestors = [
                    getattr(ancestor, "name", None)
                    for ancestor in entity_obj.ancestors()
                    if getattr(ancestor, "name", None)
                    and getattr(ancestor, "name", None) != entity_name
                ]
                depth = len(ancestors)
            except Exception:
                pass
            try:
                num_children = len(list(entity_obj.subclasses()))
            except Exception:
                pass
            try:
                num_properties = len(entity_obj.get_class_properties())
            except Exception:
                pass
        elif entity_type in ('datatype_property', 'object_property'):
            try:
                domain_list = (
                    entity_obj.domain
                    if isinstance(entity_obj.domain, list)
                    else [entity_obj.domain]
                )
                parents = [
                    getattr(d, "name", str(d))
                    for d in domain_list
                    if getattr(d, "name", None)
                ]
            except Exception:
                pass
            depth = len(parents)
            num_children = 0
            num_properties = 1

        return parents, depth, num_children, num_properties, entity_type

    def _augment_dataframe_with_ontology(self, df):
        if self.loaded_ontology is None:
            return df, None

        matcher = self._get_ontology_matcher()
        matching_pairs, entity_stats = self._get_ontology_matching_entities()
        ontology_entities = TrepanReloadedExtractor._unwrap_matching_entities(matching_pairs)
        if matcher is not None:
            matcher._log_ontology_matching_entities(entity_stats)

        if not ontology_entities:
            return df, None

        target_col = df.columns[-1]
        features_df = df.iloc[:, :-1]
        target_series = df[target_col]
        feature_names = list(features_df.columns)
        n_features = len(feature_names)
        n_rows = len(df)

        new_columns = {}
        added_columns = []
        summary = []
        mapping_log_lines = []
        detected_types = set()
        mapped_count = 0

        progress = QProgressDialog(
            "Enriqueciendo datos con ontología...",
            None,
            0,
            n_features,
            self,
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setWindowTitle("Enriquecimiento Ontológico")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        progress.show()
        QApplication.processEvents()

        try:
            for idx, feature_name in enumerate(feature_names):
                progress.setValue(idx)
                progress.setLabelText(
                    f"Procesando feature {idx + 1}/{n_features}: {feature_name}"
                )
                if idx % 5 == 0:
                    QApplication.processEvents()

                concept_obj, score, entity_type = self._find_best_ontology_match(
                    feature_name, ontology_entities
                )
                concept_name = (
                    getattr(concept_obj, "name", None) if concept_obj is not None else None
                )

                summary.append({
                    'feature': feature_name,
                    'concept': concept_name or 'UNMAPPED',
                    'matched_entity': concept_name or 'UNMAPPED',
                    'entity_type': entity_type or 'unknown',
                    'score': score,
                    'enriched': concept_obj is not None and score >= self.ontology_match_threshold,
                    'ontology_mapped': concept_obj is not None and score >= self.ontology_match_threshold,
                })

                if concept_obj is None or score < self.ontology_match_threshold:
                    if concept_name:
                        mapping_log_lines.append(
                            f"{feature_name} -> UNMAPPED [{entity_type or 'unknown'}] score={score:.2f}"
                        )
                    continue

                mapped_count += 1
                if entity_type and entity_type != 'unknown':
                    detected_types.add(entity_type)
                mapping_log_lines.append(
                    f"{feature_name} -> {concept_name} [{entity_type or 'unknown'}] score={score:.2f}"
                )
                parents, depth, num_children, num_properties, entity_type = (
                    self._collect_entity_enrichment_metadata(
                        concept_obj, concept_name, matcher
                    )
                )

                concept_col = f"onto_{feature_name}_concept"
                score_col = f"onto_{feature_name}_concept_score"
                depth_col = f"onto_{feature_name}_depth"
                parents_col = f"onto_{feature_name}_parents"
                children_col = f"onto_{feature_name}_num_children"
                prop_col = f"onto_{feature_name}_num_properties"

                new_columns[concept_col] = pd.Series(
                    [concept_name] * n_rows, index=df.index, dtype=object
                )
                new_columns[score_col] = pd.Series(
                    [round(score, 3)] * n_rows, index=df.index, dtype=float
                )
                new_columns[depth_col] = pd.Series(
                    [depth] * n_rows, index=df.index, dtype=int
                )
                new_columns[parents_col] = pd.Series(
                    [", ".join(parents) if parents else "None"] * n_rows,
                    index=df.index,
                    dtype=object,
                )
                new_columns[children_col] = pd.Series(
                    [num_children] * n_rows, index=df.index, dtype=int
                )
                new_columns[prop_col] = pd.Series(
                    [num_properties] * n_rows, index=df.index, dtype=int
                )
                added_columns.extend([
                    concept_col, score_col, depth_col, parents_col, children_col, prop_col
                ])

                if features_df[feature_name].dtype == object:
                    mapped_values, mapped_scores = self._map_series_values_to_ontology(
                        df[feature_name]
                    )
                    value_col = f"onto_{feature_name}_value_concept"
                    value_score_col = f"onto_{feature_name}_value_score"
                    new_columns[value_col] = pd.Series(
                        mapped_values, index=df.index, dtype=object
                    )
                    new_columns[value_score_col] = pd.Series(
                        mapped_scores, index=df.index, dtype=float
                    )
                    added_columns.extend([value_col, value_score_col])

            progress.setValue(n_features)
            progress.setLabelText(
                f"Completado: {mapped_count}/{n_features} features enriquecidas, "
                f"{len(added_columns)} columnas añadidas"
            )
            QApplication.processEvents()
        finally:
            progress.close()

        if new_columns:
            extra_df = pd.DataFrame(new_columns, index=df.index)
            enriched_features = pd.concat([features_df, extra_df], axis=1)
        else:
            enriched_features = features_df

        augmented_df = pd.concat([enriched_features, target_series], axis=1)

        semantic_stats = {}
        inferred_columns = []
        try:
            processor = OntologyProcessor(
                self.loaded_ontology, matcher=matcher
            )
            augmented_df, semantic_stats = processor.apply_semantic_feature_engineering(
                augmented_df,
                target_column=target_col,
                log=True,
            )
            inferred_columns = semantic_stats.get("inferred_columns", [])
            added_columns = list(dict.fromkeys(added_columns + inferred_columns))
        except Exception as eng_error:
            print(f"[WARN] Engenharia semântica avançada: {eng_error}")

        original_arff_features = n_features
        total_training = len(augmented_df.columns) - 1
        augmentation_info = {
            'added_columns': added_columns,
            'summary': summary,
            'match_threshold': self.ontology_match_threshold,
            'mapped_features': mapped_count,
            'total_features': n_features,
            'original_arff_features': original_arff_features,
            'semantic_engineering': semantic_stats,
            'inferred_columns': inferred_columns,
            'total_training_features': total_training,
        }

        if matcher is not None:
            matcher._log_feature_mapping_summary(
                feature_names, mapping_log_lines, mapped_count, detected_types
            )
        print(
            f"[INFO] Enriquecimento ontologico: {mapped_count}/{n_features} features mapeadas, "
            f"{len(added_columns)} colunas derivadas (threshold={self.ontology_match_threshold})"
        )
        if semantic_stats:
            print(
                f"[INFO] Features Originales ARFF: {semantic_stats.get('original_features', n_features)}"
            )
            print(
                f"[INFO] Features Mapeadas Diretamente: "
                f"{semantic_stats.get('mapped_direct', mapped_count)}"
            )
            print(
                f"[INFO] Total para Treinamento: "
                f"{semantic_stats.get('total_training_features', total_training)}"
            )
        if mapped_count == 0 and n_features > 0:
            print(
                "[WARN] A ontologia foi carregada, mas nenhuma feature do ARFF foi mapeada "
                "semánticamente. Verifique que las features estén representadas como owl:Class, "
                "owl:DatatypeProperty ou owl:ObjectProperty e se os nomes/labels estão "
                "alinhados com o ARFF."
            )

        return augmented_df, augmentation_info

    def _has_loaded_data(self):
        return self.original_data is not None or self.current_data is not None

    def _get_original_xy(self):
        if self.original_data is not None:
            return self.original_data
        return self.current_data

    def _get_original_feature_names(self):
        """Nomes das features ARFF (sem onto_*), agnóstico ao dataset."""
        if self.feature_names_original:
            return list(self.feature_names_original)
        meta = getattr(self, 'arff_meta', None) or {}
        names = meta.get('original_features') or meta.get('features')
        if names:
            return list(names)
        X, _ = self._get_original_xy()
        n = X.shape[1] if X is not None and hasattr(X, 'shape') else 0
        return [f"feature_{i}" for i in range(n)]

    def _get_original_feature_types(self):
        """Tipos declarados no ARFF, alinhados nominalmente às features originais."""
        names = self._get_original_feature_names()
        attributes = list((getattr(self, 'arff_meta', None) or {}).get('attributes') or [])
        by_name = {
            str(item.get('name')): item.get('type', 'numeric')
            for item in attributes
            if isinstance(item, dict) and item.get('name') in names
        }
        if len(by_name) != len(names):
            return None
        return [by_name[name] for name in names]

    def _slice_matrix_to_original_features(self, X, matrix_feature_names=None):
        """Extrai sub-matriz das colunas ARFF originais por nome (agnóstico a |features|)."""
        if X is None:
            return X
        X_arr = np.asarray(X)
        orig_names = self._get_original_feature_names()
        if matrix_feature_names is None:
            if X_arr.shape[1] == len(orig_names):
                return X_arr
            raise ValueError(
                f"Matriz com {X_arr.shape[1]} colunas sem feature_names; "
                f"esperadas {len(orig_names)} features ARFF originais. "
                "Passe matrix_feature_names para alinhamento por nome."
            )

        name_to_idx = {n: i for i, n in enumerate(matrix_feature_names)}
        indices = [name_to_idx[n] for n in orig_names if n in name_to_idx]
        if len(indices) == len(orig_names):
            return X_arr[:, indices]
        if indices:
            print(
                f"[WARN] Nem todas as features originais estão na matriz; "
                f"usando {len(indices)}/{len(orig_names)} colunas por nome."
            )
            return X_arr[:, indices]
        raise ValueError(
            f"Nenhuma feature ARFF original encontrada na matriz "
            f"({X_arr.shape[1]} colunas, {len(orig_names)} nomes esperados)."
        )

    def _get_mlp_training_xy(self):
        """Oráculo MLP: apenas features ARFF originais."""
        X, y = self._get_original_xy()
        orig_names = self._get_original_feature_names()
        X_mlp = self._slice_matrix_to_original_features(X, orig_names)
        print(
            f"[DEBUG] Treinando MLP com {X_mlp.shape[1]} features "
            f"(originais ARFF: {len(orig_names)})."
        )
        return X_mlp, y

    def _get_augmented_xy(self):
        if self.augmented_data is not None:
            return self.augmented_data
        return None

    def _encode_augmented_for_tree(self):
        """Devolve a matriz OWL criada por fit/transform no split de treino."""
        if self.X_encoded_aug is not None and self.y_encoded_aug is not None:
            return self.X_encoded_aug, self.y_encoded_aug
        aug = self._get_augmented_xy()
        if aug is None:
            return None, None
        X_aug, y_aug = aug
        feature_names_aug = self.feature_names_augmented or self._get_original_feature_names()
        orig_names = self._get_original_feature_names()
        rows = np.asarray(X_aug).tolist()
        X_enc = self.trepan.mlp_trainer.encode_columns_by_name(
            rows, feature_names_aug, orig_names
        )
        y_enc = self.trepan.label_encoder.transform(y_aug)
        print(
            f"[DEBUG] Matriz Trepan Reloaded (enriquecida): {X_enc.shape[1]} features "
            f"(ARFF={len(orig_names)}, total nomes={len(feature_names_aug)})."
        )
        return X_enc, y_enc

    def _uses_dual_data_pipeline(self):
        return self.loaded_ontology is not None and self.original_data is not None

    def _diagnose_and_harden_mlp_original(self, model):
        """Diagnóstico obrigatório + reotimização se MLP degenerado."""
        from core.mlp_diagnostic import ensure_robust_mlp_original

        eval_split = getattr(self.trepan.mlp_trainer, 'eval_split', None)
        if eval_split is None:
            return model, None

        hardened, diag = ensure_robust_mlp_original(
            eval_split['X_train'],
            eval_split['y_train'],
            eval_split['X_test'],
            eval_split['y_test'],
            existing_model=model,
        )
        self.trepan.mlp_trainer.model = hardened
        self.mlp_model = hardened
        if eval_split is not None:
            self.trepan.mlp_trainer._attach_bundle(
                hardened,
                eval_split['X_train'],
                'MLP Original',
                'original',
                X_test=eval_split['X_test'],
            )
        if diag.get('robust_retrained'):
            tm = {
                'accuracy': diag.get('accuracy'),
                'precision': diag.get('precision_weighted'),
                'recall': diag.get('recall_weighted'),
                'f1': diag.get('f1_weighted'),
                'balanced_accuracy': diag.get('balanced_accuracy'),
            }
            self.trepan.mlp_trainer.last_optimization_summary = {
                'model': hardened,
                'optimization_method': diag.get(
                    'robust_optimization_method', 'robust_retrain'
                ),
                'best_params': diag.get('robust_best_params', {}),
                'test_metrics': tm,
            }
        return hardened, diag

    def _oracle_eval_splits(self):
        residual_trainer = getattr(self.trepan, 'mlp_trainer_residual', None)
        res_split = (
            getattr(residual_trainer, 'eval_split', None) if residual_trainer else None
        )
        return {
            'original': self._eval_split_original,
            'enriched': self._eval_split_augmented,
            'augmented': self._eval_split_augmented,
            'residual': res_split,
        }

    def _bundle_for_oracle(self, oracle):
        if oracle is self.mlp_model:
            return getattr(self.trepan.mlp_trainer, 'bundle', None)
        if getattr(oracle, 'ORACLE_TYPE', None) == 'residual_ontological':
            return getattr(self.trepan.mlp_trainer_residual, 'bundle', None)
        if oracle is getattr(self, 'mlp_model_onto', None):
            return getattr(self.trepan.mlp_trainer_onto, 'bundle', None)
        return None

    def _ensure_viable_reloaded_oracle(self, mlp_original, selected_oracle, oracle_label):
        """Garante oráculo Reloaded não degenerado; fallback para MLP Original."""
        from core.mlp_diagnostic import diagnose_oracle, log_oracle_diagnostic
        from core.mlp_optimizer import train_robust_mlp_original

        splits = self._oracle_eval_splits()
        if splits.get('original') is None and splits.get('enriched') is None:
            return selected_oracle, oracle_label, None

        # Esta função pode alterar o professor/fazer fallback, portanto a sua
        # decisão NÃO pode observar o teste externo. Para a verificação de
        # degenerescência usamos somente a porção de desenvolvimento de cada
        # espaço de features. O teste permanece reservado ao relatório final.
        development_splits = {}
        for key, split in splits.items():
            if not split or split.get('X_train') is None:
                development_splits[key] = None
                continue
            development_splits[key] = {
                'X_train': split['X_train'],
                'X_test': split['X_train'],
                'y_train': split.get('y_train'),
                'y_test': split.get('y_train'),
            }

        bundle = self._bundle_for_oracle(selected_oracle)
        diag = diagnose_oracle(
            selected_oracle,
            splits=development_splits,
            oracle_label=oracle_label,
            bundle=bundle,
        )
        diag['selection_role'] = 'development_only_viability_check'
        diag['test_used_for_selection'] = False
        log_oracle_diagnostic(diag)

        if not diag.get('degenerate'):
            return selected_oracle, oracle_label, diag

        if selected_oracle is mlp_original:
            print(
                "[WARNING] MLP Original (oráculo Reloaded) degenerado — "
                "Trepan-Reloaded resultará em árvore trivial."
            )
            return selected_oracle, oracle_label, diag

        print(
            f"[INFO] Oráculo {oracle_label} degenerado — a tentar reotimizar MLP Residual..."
        )
        res_split = splits.get('residual')
        if res_split is not None:
            try:
                result = train_robust_mlp_original(
                    res_split['X_train'],
                    res_split['y_train'],
                    # Reporting argument intentionally reuses development only;
                    # train_robust_mlp_original faz a selecção no seu holdout
                    # interno e o teste externo não participa desta decisão.
                    res_split['X_train'],
                    res_split['y_train'],
                )
                residual_trainer = getattr(self.trepan, 'mlp_trainer_residual', None)
                if residual_trainer is not None:
                    residual_trainer.model = result['model']
                    if residual_trainer.eval_split is not None:
                        residual_trainer._attach_bundle(
                            result['model'],
                            res_split['X_train'],
                            "MLP Residual Ontológico",
                            "residual_ontological",
                            X_test=res_split['X_test'],
                        )
                from core.residual_ontological_oracle import ResidualOntologicalOracle
                if isinstance(selected_oracle, ResidualOntologicalOracle):
                    selected_oracle = ResidualOntologicalOracle(
                        mlp_original=selected_oracle.mlp_original,
                        mlp_residual=result['model'],
                        original_column_indices=selected_oracle.original_column_indices,
                        onto_column_indices=selected_oracle.onto_column_indices,
                        n_enriched_features=selected_oracle.n_enriched_features,
                    )
                    self.mlp_model_residual = selected_oracle
                bundle = self._bundle_for_oracle(selected_oracle)
                diag = diagnose_oracle(
                    selected_oracle,
                    splits=development_splits,
                    oracle_label=oracle_label,
                    bundle=bundle,
                )
                diag['selection_role'] = 'development_only_viability_check'
                diag['test_used_for_selection'] = False
                log_oracle_diagnostic(diag)
                if not diag.get('degenerate'):
                    return selected_oracle, oracle_label, diag
            except Exception as exc:
                print(f"[WARN] Reotimização MLP Residual falhou: {exc}")

        print(
            "[WARNING] Oráculo Reloaded continua degenerado — "
            "a usar MLP Original como fallback."
        )
        fallback_bundle = self._bundle_for_oracle(mlp_original)
        fallback_diag = diagnose_oracle(
            mlp_original,
            splits=development_splits,
            oracle_label='MLP Original',
            bundle=fallback_bundle,
        )
        fallback_diag['selection_role'] = 'development_only_viability_check'
        fallback_diag['test_used_for_selection'] = False
        log_oracle_diagnostic(fallback_diag)
        return mlp_original, 'MLP Original', fallback_diag

    def _get_trepan_reloaded_context(self, feature_names, feature_names_aug=None):
        from core.trepan_reloaded_context import get_trepan_reloaded_context

        residual_trainer = getattr(self.trepan, 'mlp_trainer_residual', None)
        eval_residual = (
            getattr(residual_trainer, 'eval_split', None) if residual_trainer else None
        )
        mlp_residual_pipeline = (
            residual_trainer.model
            if residual_trainer is not None and residual_trainer.model is not None
            else None
        )
        residual_names = None
        if residual_trainer is not None and getattr(residual_trainer, 'bundle', None):
            residual_names = list(residual_trainer.bundle.feature_names)

        ctx = get_trepan_reloaded_context(
            ontology_enabled=self.loaded_ontology is not None,
            ontology_acceptance=self.ontology_acceptance,
            selected_oracle_label=self.selected_oracle_label,
            selected_oracle=self.selected_oracle,
            mlp_original=self.mlp_model,
            mlp_onto=self.mlp_model_onto,
            mlp_residual_pipeline=mlp_residual_pipeline,
            mlp_residual_oracle=self.mlp_model_residual,
            eval_split_original=self._eval_split_original,
            eval_split_enriched=self._eval_split_augmented,
            eval_split_residual=eval_residual,
            feature_names_original=feature_names,
            feature_names_enriched=feature_names_aug or self.feature_names_augmented,
            feature_names_residual=residual_names,
            bundle_original=getattr(self.trepan.mlp_trainer, 'bundle', None),
            bundle_onto=getattr(self.trepan.mlp_trainer_onto, 'bundle', None),
            bundle_residual=(
                getattr(residual_trainer, 'bundle', None) if residual_trainer else None
            ),
        )
        ctx['mlp_original_ref'] = self.mlp_model
        return ctx

    def _rebuild_augmented_dataset(self):
        """Bloqueia o antigo enriquecimento sobre o dataset completo."""
        raise RuntimeError(
            "_rebuild_augmented_dataset foi desativado: use "
            "_prepare_ontology_train_test, que ajusta a ontologia somente no treino."
        )

    def _prepare_ontology_train_test(self):
        """Ajusta OWL no treino e transforma treino/teste sem leakage."""
        if self.loaded_ontology is None or self.original_data is None:
            return None
        split = getattr(self.trepan.mlp_trainer, 'eval_split', None) or {}
        train_indices = split.get('train_indices')
        test_indices = split.get('test_indices')
        if train_indices is None or test_indices is None:
            raise ValueError(
                "O split original não contém índices auditáveis; elimine a cache antiga e treine novamente."
            )
        X_raw, y_raw = self.original_data
        feature_names = self._get_original_feature_names()
        train_indices = np.asarray(train_indices, dtype=int)
        test_indices = np.asarray(test_indices, dtype=int)
        train_frame = pd.DataFrame(np.asarray(X_raw, dtype=object)[train_indices], columns=feature_names)
        test_frame = pd.DataFrame(np.asarray(X_raw, dtype=object)[test_indices], columns=feature_names)

        forbidden_ids = []
        for column in feature_names:
            token = self._normalize_ontology_token(column)
            if token.endswith('id') or token in {'id', 'identifier', 'recordid', 'patientid'}:
                forbidden_ids.extend(test_frame[column].astype(str).tolist())

        report = self.ontology_quality_gate.evaluate(
            feature_names,
            self.loaded_ontology,
            forbidden_test_identifiers=forbidden_ids,
            reasoner_report=self.ontology_reasoner_report,
            require_reasoner=True,
        )
        self.ontology_quality_report = report.to_dict()
        self.trepan.extractor.ontology_quality_report = self.ontology_quality_report
        self.trepan.extractor.reasoner_report = dict(self.ontology_reasoner_report or {})
        if hasattr(self.trepan.extractor, 'register_quality_matches'):
            self.trepan.extractor.register_quality_matches(feature_names, report.matches)
        if not report.accepted:
            raise ValueError(
                "Ontologia rejeitada pelo quality gate: " + " | ".join(report.issues)
            )

        transformer = OntologyProcessor(
            self.loaded_ontology,
            matcher=self._get_ontology_matcher(),
            quality_gate=self.ontology_quality_gate,
            allow_train_calibrated_bounds=False,
        )
        accepted_matches = [item for item in report.matches if item.get('accepted')]
        transformer.fit(
            train_frame,
            accepted_matches=accepted_matches,
            log=True,
        )
        train_enriched = transformer.transform(train_frame)
        test_enriched = transformer.transform(test_frame)
        if list(train_enriched.columns) != list(test_enriched.columns):
            raise ValueError("Schemas OWL de treino e teste divergiram.")
        structural_only = train_enriched.shape[1] == len(feature_names)
        if structural_only:
            # A OWL continua válida como conhecimento estrutural para o TREPAN
            # Reloaded. Apenas o braço de feature engineering do professor MLP
            # ficará sem colunas derivadas e será avaliado como tal pelo gate OOF.
            print(
                "[INFO] OWL válida sem features derivadas independentes: "
                "uso estrutural no TREPAN Reloaded permanece disponível."
            )

        # Base ARFF usa encoders fitted no treino original; derivadas são numéricas.
        n_base = len(feature_names)
        X_train_base = self.trepan.mlp_trainer._encode_features(train_frame.values)
        X_test_base = self.trepan.mlp_trainer._encode_features(test_frame.values)
        train_derived = train_enriched.iloc[:, n_base:].to_numpy(dtype=float)
        test_derived = test_enriched.iloc[:, n_base:].to_numpy(dtype=float)
        X_train_aug = np.column_stack([X_train_base, train_derived])
        X_test_aug = np.column_stack([X_test_base, test_derived])
        if not np.isfinite(X_train_aug).all() or not np.isfinite(X_test_aug).all():
            raise ValueError(
                "Features OWL contêm valores ausentes/não finitos; não serão preenchidos com zero."
            )
        names = list(train_enriched.columns)
        y_encoded = self.trepan.label_encoder.transform(y_raw)
        X_full = np.empty((len(y_raw), len(names)), dtype=float)
        X_full[train_indices] = X_train_aug
        X_full[test_indices] = X_test_aug

        self.ontology_transformer = transformer
        self.trepan.extractor.ontology_processor = transformer
        self.X_encoded_aug = X_full
        self.y_encoded_aug = y_encoded
        self.feature_names_augmented = names
        self.ontology_augmented_columns = names[n_base:]
        self.ontology_feature_summary = report.matches
        self.augmented_data = (X_full, np.asarray(y_raw))
        self._eval_split_augmented = {
            'X_train': X_train_aug,
            'X_test': X_test_aug,
            'X_train_raw': train_frame.copy(),
            'y_train': y_encoded[train_indices],
            'y_test': y_encoded[test_indices],
            'train_indices': train_indices,
            'test_indices': test_indices,
            'test_role': 'locked_final_test',
        }
        augmentation = {
            'fit_scope': 'training_only',
            'fit_rows': len(train_indices),
            'test_rows': len(test_indices),
            'added_columns': names[n_base:],
            'augmented_features': names,
            'quality_gate': self.ontology_quality_report,
            'semantic_engineering': transformer.last_engineering_stats,
            'structural_only': bool(structural_only),
            'zero_fill': False,
        }
        if self.arff_meta is not None:
            self.arff_meta['augmentation'] = augmentation
            self.arff_meta['augmented_features'] = names
            self.arff_meta['derived_attributes'] = [
                {'name': name, 'type': 'ontology-derived'} for name in names[n_base:]
            ]
        return augmentation

    def load_data_from_file(self, file_path):
        result = self._load_data_from_file_impl(file_path)
        self._on_dataset_loaded(file_path)
        return result

    def _on_dataset_loaded(self, file_path):
        """Regista identidade do dataset e atualiza o estado (nunca bloqueia o carregamento)."""
        try:
            meta = getattr(self, 'arff_meta', None) or {}
            X, y = self.original_data
            classes = list(meta.get('classes') or [])
            self.dataset_info = {
                'name': meta.get('file_name') or os.path.basename(file_path), 'path': file_path,
                'rows': int(len(y)), 'features': len(meta.get('original_features') or []),
                'classes': len(classes), 'class_names': [str(c) for c in classes], 'target': meta.get('target'),
            }
            self.dataset_fingerprint = dataset_fingerprint_of(X, y)
            if self.audit is not None:
                self.audit.on_data_loaded()
        except Exception as exc:  # a observabilidade nunca impede o carregamento
            import logging
            logging.getLogger("biuri.gui").warning("Registo do dataset falhou: %s", exc)

    def _audit_check_stale(self):
        if getattr(self, 'audit', None) is not None:
            try:
                self.audit.check_stale()
            except Exception as exc:
                import logging
                logging.getLogger("biuri.gui").warning("Verificação de stale falhou: %s", exc)

    def _load_data_from_file_impl(self, file_path):
        
        self.ontology_augmented_columns = []
        self.ontology_feature_summary = []
        if hasattr(self, "clarity_trail"):
            self.clarity_trail.reset()
        if file_path.endswith('.arff'):
            relation_name, attributes, num_instances_from_header = self._extract_arff_header_info(file_path)
            
            data, meta = arff.loadarff(file_path)
            df = pd.DataFrame(data)
            
            str_df = df.select_dtypes([object])
            str_df = str_df.stack().str.decode('utf-8').unstack()
            for col in str_df:
                df[col] = str_df[col]
            
            original_features = list(df.columns[:-1])
            target_col = df.columns[-1]

            self.original_data = (
                df.iloc[:, :-1].values.copy(),
                df.iloc[:, -1].values.copy(),
            )
            self.current_data = self.original_data
            self.trepan_original_tree = None
            self.trepan_reloaded_tree = None
            self.c45_tree = None
            self.trepan_original_audit = None
            self.trepan_reloaded_audit = None
            self.feature_names_original = original_features
            self.augmented_data = None
            self.feature_names_augmented = None

            # O enriquecimento só é ajustado depois do split, no treino.
            augmentation_info = None

            self.metrics_comparator.clear_cache()
            self.mlp_model_reloaded = None
            self.mlp_model_onto = None
            self.X_encoded_aug = None
            self.y_encoded_aug = None
            self.metrics_comparator.comparison_results = {}
            if hasattr(self, 'metrics_widget') and self.metrics_widget is not None:
                self.metrics_widget.metrics_visualizer.clear_model_data()

            derived_attributes = []
            if augmentation_info:
                added_columns = augmentation_info.get('added_columns', [])
                derived_attributes = [{'name': col, 'type': 'ontology-derived'} for col in added_columns]
                self.ontology_augmented_columns = added_columns
                self.ontology_feature_summary = augmentation_info.get('summary', [])
            else:
                self.ontology_augmented_columns = []
                self.ontology_feature_summary = []

            combined_attributes = list(attributes) if attributes else []
            if derived_attributes:
                combined_attributes = combined_attributes + derived_attributes

            class_order = parse_arff_class_order(file_path, str(target_col))
            if not class_order:
                class_order = list(df.iloc[:, -1].unique())
            self.arff_meta = {
                'file_name': os.path.basename(file_path),
                'sample_count': len(df),
                'features': original_features,
                'target': target_col,
                'classes': class_order,
                'class_order_source': 'arff_header',
                'relation_name': relation_name,
                'attributes': list(attributes) if attributes else [],
                'num_instances': len(df),
                'original_features': original_features,
                'derived_attributes': derived_attributes,
                'dual_pipeline': self._uses_dual_data_pipeline(),
            }
            if augmentation_info:
                self.arff_meta['augmentation'] = augmentation_info
            if self.feature_names_augmented:
                self.arff_meta['augmented_features'] = self.feature_names_augmented

            y_all = self.original_data[1]
            self.trepan.mlp_trainer.set_metadata(
                os.path.basename(file_path),
                original_features,
                target_col,
                class_order,
                len(y_all),
            )
            if hasattr(self.trepan.mlp_trainer, 'arff_meta') and self.trepan.mlp_trainer.arff_meta is not None:
                self.trepan.mlp_trainer.arff_meta['original_features'] = original_features
                self.trepan.mlp_trainer.arff_meta['derived_attributes'] = derived_attributes
                if augmentation_info:
                    self.trepan.mlp_trainer.arff_meta['augmentation'] = augmentation_info

            if self._uses_dual_data_pipeline() and self.augmented_data is not None:
                X_aug, y_aug = self.augmented_data
                self.mlp_trainer_reloaded.set_metadata(
                    os.path.basename(file_path) + ' (enriquecido)',
                    self.feature_names_augmented,
                    target_col,
                    class_order,
                    len(y_aug),
                )
                if self.mlp_trainer_reloaded.arff_meta is not None:
                    self.mlp_trainer_reloaded.arff_meta['original_features'] = original_features
                    self.mlp_trainer_reloaded.arff_meta['derived_attributes'] = derived_attributes
                    if augmentation_info:
                        self.mlp_trainer_reloaded.arff_meta['augmentation'] = augmentation_info
            
            self._update_metrics_visualizer_data()
            
            # Verificar se ontologia foi carregada
            ontology_status = ""
            if self._uses_dual_data_pipeline():
                ontology_status = (
                    "✅ Ontología activa: datos originales preservados; "
                    "fit OWL pendente exclusivamente no split de treino.\n\n"
                )
            elif self.loaded_ontology is not None:
                mapped_feat = 0
                total_feat = len(original_features)
                if augmentation_info:
                    mapped_feat = augmentation_info.get('mapped_features', 0)
                    total_feat = augmentation_info.get('total_features', total_feat)
                warn_unmapped = (
                    mapped_feat == 0
                    and total_feat > 0
                )
                ontology_status = "✅ Ontología cargada (sin copia enriquecida — verifique el ARFF).\n\n"
                if warn_unmapped:
                    ontology_status = (
                        "⚠️ Ontología cargada, pero ninguna feature del ARFF fue mapeada "
                        "semánticamente. Verifique que las features estén representadas como "
                        "owl:Class, owl:DatatypeProperty ou owl:ObjectProperty e se os nomes/"
                        "labels estén alineados con el ARFF.\n\n"
                    )
            else:
                ontology_status = "ℹ️ Datos cargados sin ontología (carga estándar).\n\n"
            
            info_text = "✅ ¡Datos cargados con éxito!\n\n" + ontology_status
            
            # Cabeçalho da Relação (@RELATION)
            if relation_name:
                info_text += f"📋 Encabezado de la Relación (@RELATION):\n   {relation_name}\n\n"
            else:
                info_text += "📋 Encabezado de la Relación (@RELATION):\n   (no especificado)\n\n"
            
            # Número de Instâncias (@DATA)
            info_text += f"📊 Número de Instancias (@DATA):\n   {len(df)}\n\n"
            
            # Número de Atributos (@ATTRIBUTE)
            info_text += f"📈 Número de Atributos (@ATTRIBUTE):\n   {len(attributes)}\n\n"
            
            # Lista de Atributos
            if attributes:
                info_text += "📝 Lista de Atributos:\n"
                for i, attr in enumerate(attributes, 1):
                    info_text += f"   {i}. {attr['name']} ({attr['type']})\n"
            else:
                # Fallback: usa nomes das colunas do DataFrame
                info_text += "📝 Lista de Atributos:\n"
                all_features = list(df.columns[:-1]) + [df.columns[-1]]
                for i, attr_name in enumerate(all_features, 1):
                    attr_type = 'numeric' if df[attr_name].dtype in ['int64', 'float64'] else 'nominal'
                    info_text += f"   {i}. {attr_name} ({attr_type})\n"

            if derived_attributes:
                info_text += "\n🧠 Atributos Derivados de la Ontología:\n"
                for i, attr in enumerate(derived_attributes, 1):
                    info_text += f"   {i}. {attr['name']} ({attr['type']})\n"
            
            info_text += "\n" + "="*50 + "\n\n"
            info_text += "📄 Archivo: {}\n".format(self.arff_meta['file_name'])
            info_text += "🎯 Lo que queremos predecir: {}\n".format(self.arff_meta['target'])
            info_text += "🏷️ Valores de la Clase: {}\n".format(", ".join([str(c) for c in self.arff_meta['classes']]))
            info_text += "   (Total: {} valores únicos)\n\n".format(len(self.arff_meta['classes']))
            if augmentation_info and self.ontology_feature_summary:
                mapped_n = augmentation_info.get('mapped_features', 0)
                total_n = augmentation_info.get('total_features', len(self.ontology_feature_summary))
                info_text += (
                    f"🤝 Mapeo Feature → Concepto: {mapped_n}/{total_n} enriquecidas "
                    f"(threshold {augmentation_info.get('match_threshold', 0.4)})\n"
                )
                enriched_entries = [
                    e for e in self.ontology_feature_summary if e.get('enriched')
                ]
                for entry in enriched_entries[:5]:
                    info_text += (
                        f"   • {entry['feature']} → {entry['concept']} "
                        f"(score {entry['score']:.2f})\n"
                    )
                if len(enriched_entries) > 5:
                    info_text += f"   • ... y más {len(enriched_entries) - 5} mapeadas\n"
                info_text += (
                    "\n🔍 Columnas derivadas añadidas: "
                    f"{len(self.ontology_augmented_columns)} "
                    f"(metadados + features inferidas)\n"
                )
                sem = augmentation_info.get("semantic_engineering") or {}
                if sem:
                    info_text += (
                        f"\n🧬 Ingeniería Semántica de Features:\n"
                        f"   • Features Originales ARFF: {sem.get('original_features', 0)}\n"
                        f"   • Mapeadas (Clases/Propiedades): {sem.get('mapped_direct', 0)}\n"
                        f"   • Jerárquicas inferidas: {sem.get('hierarchical_features', 0)}\n"
                        f"   • Relacionales inferidas: {sem.get('relational_features', 0)}\n"
                        f"   • Restricción de valor: {sem.get('constraint_features', 0)}\n"
                        f"   • Total para entrenamiento: {sem.get('total_training_features', 0)}\n"
                    )
                    onto_cols = [
                        c for c in (augmentation_info.get("inferred_columns") or [])
                        if str(c).startswith("onto_")
                    ]
                    for col_name in onto_cols[:8]:
                        info_text += f"   • {col_name}\n"
                    if len(onto_cols) > 8:
                        info_text += f"   • ... y más {len(onto_cols) - 8} columnas onto_*\n"
            info_text += (
                "Siguiente: entrene el modelo para obtener el oráculo opaco "
                "y el árbol sustituto que lo explica."
            )

        else:
            raise ValueError("Formato de archivo no soportado. Utilice archivos ARFF.")

        self.show_results(info_text)
        n_rows = len(self.current_data) if self.current_data is not None else 0
        onto_bit = (
            " Ontología lista para enriquecer el árbol."
            if self.loaded_ontology is not None
            else ""
        )
        self._unlock_clarity(
            "datos",
            status=(
                f"Datos listos ({n_rows} instancias).{onto_bit} "
                "Siguiente: Entrenar Modelo."
            ),
        )
        
    def load_ontology(self):
        dialog = OntologyLoadDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            file_path = dialog.file_path_edit.text()
            if file_path:
                try:
                    self.show_progress("Cargando ontología...")
                    ontology_loaded = self.load_ontology_from_file(file_path)
                    if ontology_loaded:
                        self.show_results("✅ ¡Ontología cargada con éxito!")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Error al cargar ontología: {str(e)}")
                    
    def load_ontology_from_file(self, file_path):
        try:
            from owlready2 import get_ontology, OwlReadyOntologyParsingError
            
            try:
                onto = TrepanReloadedExtractor.load_ontology_file(file_path)
                reasoner_report = run_owl_reasoner(
                    onto, engine='hermit', infer_property_values=True, debug=0
                )
                if not reasoner_report.get('executed'):
                    diagnostic = dict(reasoner_report)
                    self._clear_loaded_ontology()
                    self.last_ontology_load_diagnostic = diagnostic
                    reasoner_message = diagnostic.get('user_message') or (
                        "O reasoner OWL DL não pôde ser executado. A ontologia "
                        "não será ativada e o projeto continuará sem OWL."
                    )
                    self.show_results(
                        "⚠️ ONTOLOGÍA NO ACTIVADA\n\n"
                        + reasoner_message
                        + "\n\nNo se asumió consistencia lógica y no se generaron features OWL."
                    )
                    QMessageBox.warning(
                        self,
                        "Reasoner OWL no disponible",
                        reasoner_message,
                    )
                    return False
                if reasoner_report.get('consistent') is not True:
                    raise ValueError(
                        "Ontologia logicamente inconsistente ou com classes insatisfazíveis: "
                        + str(reasoner_report.get('unsatisfiable_classes') or [])
                    )
                abox_report = self.ontology_quality_gate.audit_abox(onto)
                if not abox_report.get('accepted'):
                    raise ValueError(
                        "ABox rejeitada: a ontologia contém instâncias de registos do dataset. "
                        + str(abox_report.get('dataset_record_individuals') or [])
                    )
                self.loaded_ontology = onto
                self.loaded_ontology_path = str(Path(file_path).resolve())
                self.ontology_reasoner_report = reasoner_report
                self.last_ontology_load_diagnostic = dict(reasoner_report)
                self._ontology_match_cache = {}
                self._ontology_matcher = None
                self.ontology_augmented_columns = []
                self.ontology_feature_summary = []
                
                self.trepan.set_ontology(onto)
                self.trepan.extractor.reasoner_report = dict(reasoner_report)
                self._apply_onto_bias_weight(self.onto_feature_bias_weight)
                self._set_onto_bias_controls_enabled(True)
                
                classes = list(onto.classes())
                props = list(onto.properties())
                import hashlib
                _owl_sha = hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
                _version_values = []
                try:
                    _version_values = list(getattr(getattr(onto, 'metadata', None), 'versionInfo', []) or [])
                except Exception:
                    _version_values = []
                _owl_version = str(_version_values[0]) if _version_values else 'não declarada'
                self.loaded_ontology_metadata = {
                    'path': self.loaded_ontology_path,
                    'file_name': os.path.basename(file_path),
                    'sha256': _owl_sha,
                    'version_info': _owl_version,
                    'class_count': len(classes),
                    'property_count': len(props),
                }
                
                hierarchies_count = 0
                datatypes_found = set()
                
                for cls in classes:
                    if hasattr(cls, 'is_a') and cls.is_a:
                        hierarchies_count += len(cls.is_a) if isinstance(cls.is_a, list) else 1
                
                for prop in props:
                    if hasattr(prop, 'range'):
                        range_val = prop.range
                        range_str = str(range_val).lower()
                        if 'integer' in range_str or 'int' in range_str:
                            datatypes_found.add('xsd:integer')
                        elif 'float' in range_str or 'double' in range_str:
                            datatypes_found.add('xsd:float')
                        elif 'boolean' in range_str:
                            datatypes_found.add('xsd:boolean')
                
                report = f"""
✅ ¡Ontología cargada con éxito!

📄 Archivo: {os.path.basename(file_path)}
🔖 VersionInfo: {_owl_version}
🔐 SHA-256: {_owl_sha}

📚 Clases ({len(classes)}):
{chr(10).join([f"• {cls.name}" for cls in classes[:10]])}
{f"... y más {len(classes)-10} clases" if len(classes) > 10 else ""}

🔗 Propiedades ({len(props)}):
{chr(10).join([f"• {prop.name}" for prop in props[:5]])}
{f"... y más {len(props)-5} propiedades" if len(props) > 5 else ""}

🏗️ Estructura ontológica:
   • Jerarquías identificadas: {hierarchies_count}
   • Tipos de datos encontrados: {len(datatypes_found)}
   {chr(10).join([f"     - {dt}" for dt in list(datatypes_found)[:5]]) if datatypes_found else "     - Ningún tipo de datos específico detectado"}

💡 Mejoras activadas:
   ✅ Emparejamiento robusto con etiquetas RDFS
   ✅ Extracción de relaciones semánticas
   ✅ Reasoner HermiT ejecutado; consistencia OWL DL verificada
   ✅ Generación de datos sintéticos inteligente
   ✅ Schema estricto: sin columnas sintéticas ni relleno con ceros

🚀 Próximo paso: Cargue sus datos y entrene el modelo para usar Trepan-Reloaded!
                """
                
                if self.original_data is not None and self.arff_meta is not None:
                    self.arff_meta['dual_pipeline'] = True
                    self.arff_meta['ontology_pending_fit'] = True

                self.show_results(report)
                return True
                
            except OwlReadyOntologyParsingError as e:
                error_msg = f"""
❌ ERROR DE SINTAXIS en el archivo OWL:
{str(e)}

Verifique:
1. Etiquetas mal formadas
2. Namespaces faltantes
3. Comillas no cerradas
4. Caracteres especiales
                """
                self._clear_loaded_ontology()
                self.show_results(error_msg)
                raise
                
        except Exception as e:
            error_msg = f"""
❌ Error inesperado:
{str(e)}

Asegúrese de que:
1. El archivo es un OWL válido
2. Tiene permiso de lectura
3. No está corrupto
            """
            self._clear_loaded_ontology()
            self.show_results(error_msg)
            raise
            
    def _get_selected_training_preset(self):
        # A GUI de produção não permite treino fora do modo científico.
        return enforce_scientific_preset(
            use_cache=self.training_cache_checkbox.isChecked()
        )

    def _trepan_training_limits(self, preset):
        structure = resolve_trepan_structure_limits(preset)
        return {
            'fidelity_target': preset.trepan_fidelity_target,
            'fidelity_early_stop': preset.trepan_fidelity_early_stop,
            'max_queries': preset.trepan_max_queries,
            'max_time_seconds': preset.trepan_max_time_seconds,
            'max_depth': structure['max_depth'],
            'max_nodes': structure['max_nodes'],
            'min_samples_leaf': preset.trepan_min_samples_leaf,
        }

    def _launch_training_worker(self, preset):
        if self._training_worker is not None and self._training_worker.isRunning():
            QMessageBox.warning(self, "Advertencia", "Ya hay un entrenamiento en curso.")
            return

        self._training_cancel_flag = False
        self._last_training_preset = getattr(preset, 'key', None)
        self._pending_failure_detail = None
        self._last_cache_info = None
        if self.audit is not None:
            self.audit.on_training_started()
        # Intervalo (0, 0): indicador de ocupado; não inventamos percentagens.
        dlg = QProgressDialog(progress_text("init"), tr("progress.cancel"), 0, 0, self)
        dlg.setWindowTitle(tr("progress.title.training"))
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(True)
        self._training_progress_dialog = dlg

        worker = TrainingWorker(self, preset)
        self._training_worker = worker

        def on_progress(stage, pct, message):
            dlg.setLabelText(progress_text(stage))

        def on_cancel():
            self._training_cancel_flag = True
            worker.request_cancel()
            dlg.setLabelText(tr("progress.cancelling"))

        dlg.canceled.connect(on_cancel)
        worker.progress.connect(on_progress)
        worker.finished_ok.connect(lambda r: self._on_training_finished(r, dlg))
        worker.failed_detail.connect(self._store_failure_detail)
        worker.failed.connect(lambda e: self._on_training_failed(e, dlg))
        dlg.show()
        worker.start()

    def _on_training_progress(self, stage, pct, message):
        if self._training_progress_dialog is not None:
            self._training_progress_dialog.setLabelText(progress_text(stage))

    def _store_failure_detail(self, message, details):
        self._pending_failure_detail = details

    def _on_training_finished(self, result, dlg):
        dlg.close()
        self._training_worker = None
        if result.get('result_text'):
            self.show_results(result['result_text'])
        if result.get('success'):
            self._refresh_dataset_split_info()
            if self.audit is not None:
                self.audit.on_training_finished()
            trees_ready = (
                self.trepan_original_tree is not None
                or self.trepan_reloaded_tree is not None
            )
            self._unlock_clarity(
                "modelo",
                status=(
                    "Modelo entrenado. El oráculo ya puede responder; "
                    "abra Herramientas avanzadas para ver el árbol."
                    if trees_ready
                    else "Modelo entrenado. Genere una explicación para leer las reglas."
                ),
                nudge_advanced=trees_ready,
            )
            if trees_ready:
                self._unlock_clarity("arbol")
            self._refresh_counterfactual_panel()
        elif self.audit is not None:
            self.audit.sm.end()
            self.audit.apply_buttons()
        perf = result.get('performance_summary')
        if perf and perf.get('stopped_by_timeout'):
            QMessageBox.information(
                self,
                "Timeout",
                "El entrenamiento alcanzó el límite de tiempo configurado. "
                "Se mantuvo el mejor modelo encontrado hasta el momento.",
            )

    def _on_training_failed(self, error_msg, dlg):
        dlg.close()
        self._training_worker = None
        details, self._pending_failure_detail = self._pending_failure_detail, None
        if "cancelado" in error_msg.lower():
            self.show_results(f"⚠️ {error_msg}")
            if self.audit is not None:
                self.audit.sm.end()
                self.audit.apply_buttons()
        elif self.audit is not None:
            msg = self.audit.on_failure(error_msg, what_key="error.training", where_key="error.training_where",
                                        action_key="error.action.training", details=details)
            self._show_structured_error(msg)
            self.content_tabs.setCurrentWidget(self.audit_panel)
        else:
            QMessageBox.critical(self, "Erro", f"Erro ao treinar o modelo: {error_msg}")

    def _show_structured_error(self, msg):
        """Diálogo de erro: o que falhou, onde, ação sugerida e ID; traceback em detalhes expansíveis."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Icon.Critical)
        box.setWindowTitle(tr("error.title"))
        box.setText(msg.text)
        box.setInformativeText(
            f"{tr('error.where')}: {msg.where or '-'}\n{tr('error.action')}: {msg.action or '-'}\n"
            f"{tr('error.id')}: {msg.experiment_id or '-'}\n{tr('error.log_hint')}")
        if msg.details:
            box.setDetailedText(msg.details)
        box.exec()

    def _refresh_dataset_split_info(self):
        """Contagens do split (apenas tamanhos já definidos pelo treino; nada é recalculado)."""
        split = getattr(self, '_eval_split_original', None) or {}
        try:
            if split.get('X_train') is not None and split.get('X_test') is not None:
                self.dataset_info['train_rows'] = int(len(split['X_train']))
                self.dataset_info['test_rows'] = int(len(split['X_test']))
        except Exception:
            pass

    def _has_trained_models_for_cf(self):
        return (
            self.mlp_model is not None
            and self.trepan_original_tree is not None
            and self.trepan_reloaded_tree is not None
            and self.X_encoded is not None
            and self.y_encoded is not None
        )

    def _refresh_counterfactual_panel(self):
        """Sincroniza a aba contrafactual sem alterar o pipeline treinado."""
        if not hasattr(self, 'counterfactual_tab'):
            return
        available = []
        if self.mlp_model is not None:
            available.append("MLP Original")
        if (
            self.mlp_model_onto is not None
            and bool((self.ontology_acceptance or {}).get('accepted'))
        ):
            available.append("MLP Ontológica")
        if self.trepan_original_tree is not None:
            available.append("Trepan Original")
        if self.trepan_reloaded_tree is not None:
            available.append("Trepan Reloaded")
        if self.c45_tree is not None:
            available.append("C4.5-Nativo")

        trainer = self.trepan.mlp_trainer
        classes = list(getattr(trainer.label_encoder, 'classes_', []))
        class_labels = {index: str(label) for index, label in enumerate(classes)}
        sizes = [
            len(matrix) for matrix in (self.X_encoded, self.X_encoded_aug)
            if matrix is not None
        ]
        self.counterfactual_tab.configure(
            n_instances=min(sizes, default=0),
            class_labels=class_labels,
            available_models=available,
            dataset_name=(self.arff_meta or {}).get('file_name', 'dataset carregado'),
        )

    def _cf_oracle_predictor(self, oracle):
        """Objeto com predict/predict_proba alinhado ao espaço codificado da sessão CF."""
        trainer = self.trepan.mlp_trainer
        if getattr(oracle, "ORACLE_TYPE", None) == "residual_ontological":
            return oracle
        if oracle is self.mlp_model or oracle is getattr(trainer, "model", None):
            trainer.bypass_preprocessing = True
            return trainer
        residual_trainer = getattr(self.trepan, "mlp_trainer_residual", None)
        if residual_trainer is not None and oracle is getattr(residual_trainer, "model", None):
            residual_trainer.bypass_preprocessing = True
            return residual_trainer
        if hasattr(oracle, "predict"):
            return oracle
        trainer.bypass_preprocessing = True
        return trainer

    def _resolve_cf_feature_space(self, oracle):
        """Escolhe matriz/nomes de features conforme o oráculo activo (não só ontologia carregada)."""
        from core.feature_alignment import oracle_n_features
        from core.model_bundle import is_residual_oracle, select_split_for_oracle

        bundle = self._bundle_for_oracle(oracle)
        n_oracle = oracle_n_features(oracle)
        orig_names = self.feature_names_original
        if orig_names is None and self.arff_meta:
            orig_names = list(self.arff_meta.get("features") or [])

        splits = self._oracle_eval_splits()
        split = select_split_for_oracle(oracle, splits, bundle=bundle)

        use_augmented = False
        if self.X_encoded_aug is not None:
            if n_oracle is not None and int(self.X_encoded_aug.shape[1]) == int(n_oracle):
                use_augmented = True
            elif is_residual_oracle(oracle):
                use_augmented = True

        if use_augmented:
            X_train = self.X_encoded_aug
            y_train = (
                self.y_encoded_aug if self.y_encoded_aug is not None else self.y_encoded
            )
            feat_names = list(self.feature_names_augmented or orig_names or [])
            X_test = split.get("X_test") if split else None
            y_test = split.get("y_test") if split else None
            if X_test is None and self._eval_split_augmented:
                X_test = self._eval_split_augmented.get("X_test")
                y_test = self._eval_split_augmented.get("y_test")
        else:
            X_train = self.X_encoded
            y_train = self.y_encoded
            feat_names = list(orig_names or [])
            if split is not None:
                X_test = split.get("X_test")
                y_test = split.get("y_test")
            elif self._eval_split_original is not None:
                X_test = self._eval_split_original.get("X_test")
                y_test = self._eval_split_original.get("y_test")
            elif self.trepan.mlp_trainer.eval_split:
                X_test = self.trepan.mlp_trainer.eval_split.get("X_test")
                y_test = self.trepan.mlp_trainer.eval_split.get("y_test")
            else:
                X_test = None
                y_test = None

        return X_train, y_train, X_test, y_test, feat_names

    def _build_cf_session(self):
        if not self._has_trained_models_for_cf():
            raise ValueError("Entrene el modelo antes de generar contrafactuais.")

        oracle = self.selected_oracle or self.mlp_model
        cf_oracle = self._cf_oracle_predictor(oracle)
        X_train, y_train, X_test, y_test, feat_names = self._resolve_cf_feature_space(oracle)
        trainer = self.trepan.mlp_trainer

        from core.feature_alignment import oracle_n_features
        n_oracle = oracle_n_features(oracle)
        if n_oracle is not None and int(X_train.shape[1]) != int(n_oracle):
            raise ValueError(
                f"Desalineación de features: el oráculo espera {n_oracle} features, "
                f"pero los datos de contrafactuales tienen {X_train.shape[1]}. "
                f"Oráculo activo: {self.selected_oracle_label or 'MLP Original'}."
            )

        class_labels = {
            i: str(c) for i, c in enumerate(getattr(trainer.label_encoder, 'classes_', []))
        }
        from counterfactuals.dataset_config import build_config_from_arff_meta
        cf_config = build_config_from_arff_meta(self.arff_meta or {}, feat_names, [])

        original_names = list(self.feature_names_original or self._get_original_feature_names())
        augmented_names = list(self.feature_names_augmented or original_names)
        reloaded_names = list(self.trepan_reloaded_feature_names or original_names)
        ontology_accepted = bool((self.ontology_acceptance or {}).get('accepted'))
        onto_oracle = None
        if ontology_accepted and self.mlp_model_onto is not None:
            onto_oracle = self._cf_oracle_predictor(self.mlp_model_onto)

        tree_b_reference = self.X_encoded
        if (
            self.X_encoded_aug is not None
            and len(reloaded_names) == int(self.X_encoded_aug.shape[1])
        ):
            tree_b_reference = self.X_encoded_aug

        ontology_mapping = {
            str(item.get('feature')): item.get('concept')
            for item in (self.ontology_feature_summary or [])
            if item.get('ontology_mapped') and item.get('feature') and item.get('concept')
        }

        # Contextos exclusivos da opção «Melhorar árvore substituta». Apenas a
        # partição de desenvolvimento é exposta; o teste bloqueado não entra no
        # worker nem pode ser usado por engano durante a busca.
        improvement_contexts = {}
        original_split = self._eval_split_original or getattr(trainer, 'eval_split', None)
        if original_split is not None:
            preset = self._get_selected_training_preset()
            improvement_contexts['trepan_original'] = {
                'display_name': 'TREPAN Original',
                'oracle': self._cf_oracle_predictor(self.mlp_model),
                'tree': self.trepan_original_tree,
                'X_development': np.asarray(original_split['X_train'], dtype=float),
                'y_development': np.asarray(original_split['y_train']),
                'feature_names': list(original_names),
                'feature_space': 'original_space',
                'teacher_id': 'mlp_original',
                'teacher_label': 'MLP Original',
                'schema_hash': stable_sha256_strings(original_names),
                'ontology_hash': 'no_ontology',
                'dataset_hash': compute_dataset_hash(np.asarray(original_split['X_train']), np.asarray(original_split['y_train']), 'cf_trepan_original_development'),
                'seed': 42,
                'ontology_active': False,
                'mirrors_original': False,
                'training_limits': self._trepan_training_limits(preset),
                'locked_test_available': bool(original_split.get('X_test') is not None),
            }
            try:
                reloaded_context = self._get_trepan_reloaded_context(
                    original_names, augmented_names
                )
            except Exception as exc:
                reloaded_context = None
                print(f"[WARN] Contexto de melhoria Reloaded indisponível: {exc}")
            if reloaded_context is not None:
                reloaded_development_context = dict(reloaded_context)
                reloaded_development_context.pop('X_test', None)
                reloaded_development_context.pop('y_test', None)
                reloaded_development_context['test_role'] = 'not_exposed_to_improvement_worker'
                ontology_active_for_improvement = bool(
                    ontology_accepted
                    and self.loaded_ontology is not None
                    and reloaded_context.get('feature_space') == 'enriched'
                )
                improvement_contexts['trepan_reloaded'] = {
                    'display_name': 'TREPAN Reloaded',
                    'oracle': reloaded_context.get('oracle'),
                    'tree': self.trepan_reloaded_tree,
                    'X_development': np.asarray(
                        reloaded_context['X_train'], dtype=float
                    ),
                    'y_development': np.asarray(reloaded_context['y_train']),
                    'feature_names': list(reloaded_context['feature_names']),
                    'original_feature_names': list(original_names),
                    'feature_space': 'ontological_space' if ontology_active_for_improvement else 'original_space',
                    'teacher_id': 'mlp_ontological' if ontology_active_for_improvement else 'mlp_original_fallback',
                    'teacher_label': 'MLP Ontológico/Híbrido' if ontology_active_for_improvement else 'MLP Original (fallback)',
                    'schema_hash': stable_sha256_strings(list(reloaded_context['feature_names'])),
                    'ontology_hash': compute_ontology_hash(self.loaded_ontology_path, self.loaded_ontology) if ontology_active_for_improvement else 'no_ontology',
                    'dataset_hash': compute_dataset_hash(np.asarray(reloaded_context['X_train']), np.asarray(reloaded_context['y_train']), 'cf_trepan_reloaded_development'),
                    'seed': 42,
                    'ontology_active': ontology_active_for_improvement,
                    'mirrors_original': bool(
                        self.trepan_reloaded_tree is self.trepan_original_tree
                        or not ontology_active_for_improvement
                    ),
                    'reloaded_context': reloaded_development_context,
                    'ontology': self.loaded_ontology,
                    'ontology_quality_report': self.ontology_quality_report,
                    'reasoner_report': self.ontology_reasoner_report,
                    'semantic_rules': list(
                        getattr(self.trepan.extractor, 'semantic_rules', []) or []
                    ),
                    'onto_feature_bias_weight': self.onto_feature_bias_weight,
                    'c45_baseline': self.c45_tree,
                    'training_limits': dict(
                        getattr(self.trepan.extractor, '_training_limits', {}) or {}
                    ),
                    'locked_test_available': bool(
                        reloaded_context.get('X_test') is not None
                    ),
                }

        # --- Contexto para o pipeline cfkit (Parte 5/6/9/10): só TREINO, categóricas legíveis, derivadas recalculáveis
        feature_encoders_info = {}
        try:
            for position, encoder in (getattr(trainer, 'feature_encoders', {}) or {}).items():
                if int(position) < len(original_names):
                    feature_encoders_info[str(original_names[int(position)])] = {
                        'labels': [str(c) for c in getattr(encoder, 'classes_', [])]
                    }
        except Exception as exc:  # metadados em falta nunca bloqueiam a explicação
            print(f"[WARN] encoders de categóricas indisponíveis para CF: {exc}")
        cf_train_original = (
            np.asarray(original_split['X_train'], dtype=float) if original_split is not None and original_split.get('X_train') is not None else None
        )
        aug_split = self._eval_split_augmented
        cf_train_augmented = (
            np.asarray(aug_split['X_train'], dtype=float) if aug_split is not None and aug_split.get('X_train') is not None else None
        )
        augmented_derive_fn = None
        try:
            processor = getattr(self.trepan.extractor, 'ontology_processor', None)
            if (processor is not None and getattr(processor, 'is_fitted_', False)
                    and augmented_names and all(n in augmented_names for n in original_names)):
                positions = [augmented_names.index(n) for n in original_names]

                def augmented_derive_fn(rows, _p=processor, _pos=positions, _names=list(original_names)):
                    base = np.asarray(rows, dtype=float)[:, _pos]
                    transformed, _ = _p.transform_matrix(base, _names)
                    return np.asarray(transformed, dtype=float)
        except Exception as exc:
            print(f"[WARN] derive_fn OWL indisponível para CF: {exc}")
            augmented_derive_fn = None

        return {
            'feature_encoders_info': feature_encoders_info,
            'X_cf_train_original': cf_train_original,
            'X_cf_train_augmented': cf_train_augmented,
            'augmented_derive_fn': augmented_derive_fn,
            'ontology_path': self.loaded_ontology_path,
            'active_dataset': (self.arff_meta or {}).get('file_name', 'gui'),
            'mlp_oracle': cf_oracle,
            'mlp_original': self._cf_oracle_predictor(self.mlp_model),
            'mlp_onto': onto_oracle,
            'X_train_enc': X_train,
            'y_train_enc': y_train,
            'X_test_enc': X_test,
            'y_test_enc': y_test,
            'tree_a': self.trepan_original_tree,
            'tree_b': self.trepan_reloaded_tree,
            'c45_tree': self.c45_tree,
            'transformed_feature_names': feat_names,
            'X_train_original': self.X_encoded,
            'y_train_original': self.y_encoded,
            'feature_names_original': original_names,
            'X_train_augmented': self.X_encoded_aug,
            'y_train_augmented': self.y_encoded_aug,
            'feature_names_augmented': augmented_names,
            'tree_a_feature_names': original_names,
            'tree_b_feature_names': reloaded_names,
            'X_tree_a_reference': self.X_encoded,
            'X_tree_b_reference': tree_b_reference,
            'class_labels': class_labels,
            'class_names': list(class_labels.values()),
            'is_multiclass': len(class_labels) > 2,
            'config': cf_config,
            'dataset_name': (self.arff_meta or {}).get('file_name', 'gui'),
            'feature_intervals': trainer.get_feature_intervals(X_train),
            'constraints': cf_config,
            'ontology': self.loaded_ontology,
            'ontology_acceptance': self.ontology_acceptance,
            'ontology_feature_mapping': ontology_mapping,
            'improvement_contexts': improvement_contexts,
            'improvement_sample_size': 1200,
        }

    def _ensure_counterfactual_modules_loaded(self):
        """TensorFlow/CLEAR must init on the GUI thread, not inside QThread.run()."""
        if self._counterfactual_modules_loaded:
            return
        from counterfactuals.service import (  # noqa: F401
            build_counterfactual_tree_from_session,
            evaluate_transfer_from_session,
            generate_explanation_from_session,
            generate_global_counterfactuals_from_session,
            generate_counterfactuals_from_session,
            improve_surrogates_from_session,
        )
        self._counterfactual_modules_loaded = True

    def _launch_cf_worker(self, mode, cf_result=None, options=None):
        if self._warn_if_busy("ejecutar contrafactuales"):
            return
        try:
            session = self._build_cf_session()
        except ValueError as exc:
            QMessageBox.warning(self, "Advertencia", str(exc))
            return

        self._ensure_counterfactual_modules_loaded()
        if hasattr(self, 'counterfactual_tab'):
            self.counterfactual_tab.set_busy(True)

        dlg = QProgressDialog("Procesando contrafactuales...", "Cancelar", 0, 100, self)
        dlg.setWindowTitle("Contrafactuales")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setValue(0)
        self._cf_progress_dialog = dlg

        worker = CounterfactualWorker(
            session,
            mode=mode,
            cf_result=cf_result,
            options=options,
            parent=self,
        )
        self._cf_worker = worker

        worker.progress.connect(self._on_cf_progress)
        worker.finished_ok.connect(lambda result: self._on_cf_finished(result, dlg, mode))
        worker.failed.connect(lambda err: self._on_cf_failed(err, dlg))
        dlg.canceled.connect(worker.request_cancel)
        dlg.show()
        worker.start()

    def _on_cf_progress(self, stage, pct, message):
        dlg = self._cf_progress_dialog
        if dlg is None:
            return
        dlg.setValue(max(0, min(100, pct)))
        if self._cf_progress_dialog is dlg:
            dlg.setLabelText(message)

    def _disconnect_cf_worker(self):
        worker = self._cf_worker
        if worker is not None:
            try:
                worker.progress.disconnect(self._on_cf_progress)
            except TypeError:
                pass
        self._cf_worker = None
        self._cf_progress_dialog = None

    def _on_cf_finished(self, result, dlg, mode):
        dlg.close()
        self._cf_progress_dialog = None
        if hasattr(self, 'counterfactual_tab'):
            self.counterfactual_tab.set_busy(False)
        worker = self._cf_worker
        if worker is not None:
            worker.finished.connect(
                lambda: self._disconnect_cf_worker(),
                Qt.ConnectionType.SingleShotConnection,
            )

        if 'counterfactuals' in result and result['counterfactuals'] is not None:
            generated = result['counterfactuals']
            if generated.get('consistency') is not None:
                self.cf_result = generated
            else:
                result_type = generated.get('result_type')
                if result_type == 'global_rules':
                    self.cf_global_result = generated
                elif result_type == 'counterfactual_tree':
                    self.cf_tree_result = generated
                else:
                    self.cf_interactive_result = generated
                self.counterfactual_tab.display_result(generated)
        if 'transfer' in result:
            self.cf_transfer_result = result['transfer']
            self.counterfactual_tab.display_result(self.cf_transfer_result)
        if 'improve' in result:
            self.cf_improve_result = result['improve']
            self.trepan_improved_tree = result['improve'].get('trepan_improved')
            self.trepan_reloaded_improved_tree = result['improve'].get('trepan_reloaded_improved')

        lines = ["Contrafactuales procesados.\n"]
        if 'counterfactuals' in result and result['counterfactuals'] is not None:
            lines.append(result['counterfactuals'].get('narrative', '') + "\n")
        if self.cf_transfer_result and 'transfer' in result:
            lines.append(self.cf_transfer_result.get('interpretation', '') + "\n")
        if self.cf_result and self.cf_result.get('consistency'):
            lines.append("Consistencia (A/B/C):\n")
            for row in self.cf_result['consistency']:
                a, b, c = row['indicador_a'], row['indicador_b'], row['indicador_c']
                if a == a:
                    lines.append(
                        f"  {row['metodo_CF']} × {row['tipo_sustituto']}: "
                        f"A={a:.3f} B={b:.3f} C={c:.3f}\n"
                    )
                else:
                    lines.append(
                        f"  {row['metodo_CF']} × {row['tipo_sustituto']}: sin CFs válidos\n"
                    )
        if self.cf_improve_result:
            lines.append(
                "\nMELHORIA CONTROLADA DAS ÁRVORES SUBSTITUTAS\n"
                "Partições: ajuste × seleção × auditoria interna independente; "
                "teste final não consultado.\n"
                "Comparação causal do gate: reextração sem CF × reextração com CF, "
                "com a mesma semente.\n"
            )
            for item in self.cf_improve_result.get('results', {}).values():
                status = item.get('status', 'desconhecido')
                if status == 'mirrored_no_ontology':
                    decision = "ESPELHADA (sem OWL aceite)"
                elif item.get('accepted'):
                    decision = "APROVADA PELO QUALITY GATE"
                else:
                    decision = "REJEITADA; ÁRVORE ATUAL PRESERVADA"
                lines.append(f"\n{item.get('model_name', 'TREPAN')} — {decision}\n")
                before = item.get('baseline_metrics') or {}
                control = item.get('matched_no_cf_metrics') or {}
                after = item.get('candidate_metrics') or {}
                if after:
                    lines.append(
                        "  Balanced accuracy: "
                        f"atual*={before.get('balanced_accuracy', 0):.1%} | "
                        f"sem CF={control.get('balanced_accuracy', 0):.1%} | "
                        f"com CF={after.get('balanced_accuracy', 0):.1%}\n"
                    )
                    lines.append(
                        "  Macro-F1: "
                        f"atual*={before.get('macro_f1', 0):.1%} | "
                        f"sem CF={control.get('macro_f1', 0):.1%} | "
                        f"com CF={after.get('macro_f1', 0):.1%}\n"
                    )
                    lines.append(
                        "  Fidelidade ao professor: "
                        f"atual*={before.get('fidelity', 0):.1%} | "
                        f"sem CF={control.get('fidelity', 0):.1%} | "
                        f"com CF={after.get('fidelity', 0):.1%}\n"
                    )
                augmentation = item.get('selected_augmentation') or {}
                used_by_type = augmentation.get('used_by_type') or {}
                if augmentation:
                    lines.append(
                        f"  CFs usados={augmentation.get('used', 0)} "
                        f"(A={used_by_type.get('A', 0)}, B={used_by_type.get('B', 0)}, "
                        f"C={used_by_type.get('C', 0)}); "
                        f"rejeitados por densidade={augmentation.get('density_rejected', 0)}\n"
                    )
                blockers = (item.get('acceptance_gate') or {}).get('blockers') or []
                if blockers:
                    lines.append("  Bloqueios: " + ", ".join(blockers) + "\n")
            lines.append(
                "\n* A árvore atual pode já ter visto todo o desenvolvimento e é apenas "
                "descritiva; o quality gate usa exclusivamente o par sem CF/com CF "
                "no holdout de auditoria. A árvore aprovada é reajustada no "
                "desenvolvimento completo sem abrir o teste confirmatório.\n"
            )
        keep_counterfactual_tab = mode in {
            CounterfactualWorker.STAGE_GENERATE,
            CounterfactualWorker.STAGE_GLOBAL,
            CounterfactualWorker.STAGE_TREE,
            CounterfactualWorker.STAGE_TRANSFER,
        }
        self.show_results(''.join(lines), switch_tab=not keep_counterfactual_tab)
        if keep_counterfactual_tab:
            self.content_tabs.setCurrentWidget(self.counterfactual_tab)
            self._set_status("Análise contrafactual concluída — dataset activo.")

    def _on_cf_failed(self, error_msg, dlg):
        dlg.close()
        self._cf_progress_dialog = None
        if hasattr(self, 'counterfactual_tab'):
            self.counterfactual_tab.set_busy(False)
        worker = self._cf_worker
        if worker is not None:
            worker.finished.connect(
                lambda: self._disconnect_cf_worker(),
                Qt.ConnectionType.SingleShotConnection,
            )
        if 'cancelad' in error_msg.lower():
            self.show_results(f"⚠️ {error_msg}")
        else:
            QMessageBox.critical(self, "Error", f"Error en contrafactuales: {error_msg}")

    def generate_counterfactuals_action(self):
        self._launch_cf_worker(CounterfactualWorker.STAGE_GENERATE)

    def _generate_cf_from_panel(self, options):
        self.content_tabs.setCurrentWidget(self.counterfactual_tab)
        self._launch_cf_worker(
            CounterfactualWorker.STAGE_GENERATE,
            options=dict(options or {}),
        )

    def _generate_global_cf_from_panel(self, options):
        self.content_tabs.setCurrentWidget(self.counterfactual_tab)
        self._launch_cf_worker(
            CounterfactualWorker.STAGE_GLOBAL,
            options=dict(options or {}),
        )

    def _build_cf_tree_from_panel(self, options):
        if not self.cf_interactive_result:
            QMessageBox.warning(
                self, "Advertencia",
                "Gere primeiro contrafactuais locais válidos para a instância activa.",
            )
            return
        self.content_tabs.setCurrentWidget(self.counterfactual_tab)
        self._launch_cf_worker(
            CounterfactualWorker.STAGE_TREE,
            cf_result=self.cf_interactive_result,
            options=dict(options or {}),
        )

    def _visualize_cf_tree(self, result):
        # A cópia mantida pela janela é a fonte canónica: contém o estimador
        # sklearn ajustado mesmo que o painel tenha sido redesenhado entretanto.
        result = self.cf_tree_result or result or {}
        tree = result.get('_runtime_tree_model')
        if tree is None or getattr(tree, 'tree_', None) is None:
            QMessageBox.warning(
                self, "Advertencia", "Nenhuma árvore contrafactual está disponível."
            )
            return
        try:
            viz = TreeVisualizationWidget(
                tree_model=tree,
                feature_names=result.get('feature_names') or [],
                class_names=result.get('class_names') or [],
                dataset_name=result.get('dataset', 'dataset_carregado'),
                primary_tree_label="Árvore contrafactual",
            )
            layout = self.visualization_tab.layout()
            if layout:
                for index in reversed(range(layout.count())):
                    widget = layout.itemAt(index).widget()
                    if widget is not None:
                        widget.setParent(None)
                layout.addWidget(viz)
            self.tree_widget = viz
            self.content_tabs.setCurrentWidget(self.visualization_tab)
            self._set_status("Árvore contrafactual — aba Visualização.")
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Error al visualizar árbol contrafactual: {exc}"
            )

    def _evaluate_cf_transfer_from_panel(self, options):
        self.content_tabs.setCurrentWidget(self.counterfactual_tab)
        self._launch_cf_worker(
            CounterfactualWorker.STAGE_TRANSFER,
            options=dict(options or {}),
        )

    def _export_cf_from_panel(self, result):
        if not result:
            QMessageBox.warning(
                self, "Advertencia", "No existe un resultado contrafactual para exportar."
            )
            return
        directory = QFileDialog.getExistingDirectory(
            self, "Seleccione la carpeta para exportar contrafactuales"
        )
        if not directory:
            return
        try:
            from counterfactuals.service import export_counterfactual_result
            paths = export_counterfactual_result(result, directory)
        except Exception as exc:
            QMessageBox.critical(
                self, "Error", f"Error al exportar contrafactuales: {exc}"
            )
            return
        QMessageBox.information(
            self,
            "Exportación completada",
            "Archivos creados:\n" + "\n".join(paths.values()),
        )

    def improve_surrogate_action(self):
        if not self._has_trained_models_for_cf():
            QMessageBox.warning(
                self, "Advertencia",
                "Treine as árvores antes de executar a melhoria contrafactual.",
            )
            return
        structural_ontology = bool(
            (self.ontology_acceptance or {}).get('ontology_structural_available')
            or (self.ontology_quality_report or {}).get('accepted')
        )
        dialog = SurrogateImprovementDialog(
            ontology_active=structural_ontology,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._launch_cf_worker(
            CounterfactualWorker.STAGE_IMPROVE,
            cf_result=self.cf_result,
            options=dialog.options(),
        )

    def train_model(self):
        if not self._has_loaded_data():
            QMessageBox.warning(self, "Advertencia", "¡Cargue datos primero!")
            return
        if self._warn_if_busy("entrenar"):
            return
        preset = self._get_selected_training_preset()
        self._launch_training_worker(preset)

    def _execute_training_pipeline(self, preset, progress_fn=None, cancel_fn=None):
        # Defesa em profundidade: mesmo chamadas programáticas são normalizadas
        # para o único modo permitido em produção.
        preset = enforce_scientific_preset(preset)
        if not self._has_loaded_data():
            raise ValueError("Carregue os dados primeiro.")

        def progress(stage, pct, msg):
            if progress_fn:
                progress_fn(stage, pct, msg)

        def cancelled():
            return bool(cancel_fn and cancel_fn())

        if cancelled():
            raise InterruptedError("Entrenamiento cancelado.")

        try:
            self.metrics_comparator.clear_cache()
            self.metrics_comparator.comparison_results = {}
            if hasattr(self, 'metrics_widget') and self.metrics_widget is not None:
                self.metrics_widget.metrics_visualizer.clear_model_data()

            progress("init", 2, "Preparando entrenamiento...")
            self.trepan.extractor.apply_training_preset(preset)

            self._apply_onto_bias_weight()
            
            X_mlp, y = self._get_mlp_training_xy()
            dual_pipeline = self._uses_dual_data_pipeline()
            has_ontology = self.loaded_ontology is not None

            dataset_name = 'unknown'
            if self.arff_meta is not None:
                dataset_name = self.arff_meta.get('file_name', 'unknown')
            elif self.trepan.mlp_trainer.arff_meta is not None:
                dataset_name = self.trepan.mlp_trainer.arff_meta.get('file_name', 'unknown')

            perf = PerformanceLogger(dataset_name, preset.key)
            dataset_hash = compute_dataset_hash(np.asarray(X_mlp), np.asarray(y), dataset_name)
            ontology_hash = compute_ontology_hash(
                getattr(self, 'loaded_ontology_path', None),
                self.loaded_ontology,
            )

            mode_msg = (
                f"⚙️ Modo: {preset.display_name}\n\n"
            )
            if dual_pipeline:
                mode_msg += (
                    "🧠 Pipeline doble: MLP Original + MLP Residual Ontológico; "
                    "criterio de aceptación ontológica activo.\n\n"
                )
            elif has_ontology:
                mode_msg += "🧠 Entrenamiento con ontología (datos únicos).\n\n"
            else:
                mode_msg += "🧠 Modo compatibilidad (sin ontología).\n\n"

            self.trepan.mlp_trainer.reset(keep_label_encoder=False)
            self.mlp_model_reloaded = None
            self.mlp_model_onto = None
            self.mlp_model_residual = None
            self.selected_oracle = None
            self.selected_oracle_label = None
            self.ontology_acceptance = None
            self.X_encoded_aug = None
            self.y_encoded_aug = None
            self._eval_split_original = None
            self._eval_split_augmented = None
            self.mlp_original_diagnostic = None
            self.reloaded_oracle_diagnostic = None
            self.trepan_original_audit = None
            self.trepan_reloaded_audit = None
            self.cf_improve_result = None
            self.trepan_improved_tree = None
            self.trepan_reloaded_improved_tree = None

            model = None
            X_encoded = None
            y_encoded = None
            message = ""

            cache_key = None
            self._last_cache_info = {'used': False, 'key': None}
            orig_feature_names = self._get_original_feature_names()
            n_orig_features = len(orig_feature_names)
            if preset.use_cache and not has_ontology:
                cache_key = build_cache_key(
                    model_role='mlp_original',
                    dataset_hash=dataset_hash,
                    ontology_hash=ontology_hash,
                    preprocessing_hash=compute_preprocessing_hash(
                        orig_feature_names, n_orig_features, False
                    ),
                    preset=preset,
                    feature_space='original',
                    n_features=n_orig_features,
                    feature_names=orig_feature_names,
                    training_mode=preset.key,
                )
                cached = load_cached_training(
                    cache_key,
                    expected_n_features=n_orig_features,
                    expected_feature_space='original',
                    feature_names=orig_feature_names,
                )
                if cached is not None:
                    self._last_cache_info = {'used': True, 'key': cache_key}
                    t_cache = perf.start_stage("MLP Original (cache)")
                    self.trepan.mlp_trainer.model = cached['model']
                    restore_mlp_trainer_state(self.trepan.mlp_trainer, cached['mlp_trainer_state'])
                    self.trepan.mlp_trainer.eval_split = cached.get('eval_split')
                    self.trepan.mlp_trainer.bundle = cached.get('bundle')
                    self.mlp_model = cached['model']
                    X_encoded = cached['X_encoded']
                    y_encoded = cached['y_encoded']
                    message = "MLP cargado de la caché."
                    from core.model_bundle import log_model_input_check
                    if self.trepan.mlp_trainer.bundle is not None and self.trepan.mlp_trainer.eval_split:
                        log_model_input_check(
                            self.trepan.mlp_trainer.bundle,
                            self.trepan.mlp_trainer.eval_split['X_train'],
                            using_cache=True,
                            operation="predict",
                        )
                    perf.end_stage(
                        "MLP Original (cache)",
                        t_cache,
                        n_features=X_encoded.shape[1] if X_encoded is not None else None,
                        used_cache=True,
                        best_score=(cached.get('metrics') or {}).get('accuracy'),
                    )
                    progress("mlp", 35, "MLP Original cargado de la caché")

            if model is None and self.mlp_model is None:
                progress("mlp", 8, "Entrenando MLP Original...")
                t_mlp = perf.start_stage("MLP Original")
                message, model, X_encoded, y_encoded = self.trepan.train_mlp(
                    X_mlp, y,
                    dataset_name=dataset_name,
                    preset=preset,
                    cancel_fn=cancel_fn,
                    progress_fn=progress_fn,
                )
                if cancelled():
                    raise InterruptedError("Entrenamiento cancelado.")
                if model is None:
                    return {'success': False, 'result_text': f"❌ Error en el entrenamiento: {message}"}
                summary = self.trepan.mlp_trainer.last_optimization_summary or {}
                perf.end_stage(
                    "MLP Original",
                    t_mlp,
                    n_features=X_encoded.shape[1],
                    n_trials=summary.get('n_trials'),
                    best_score=(summary.get('test_metrics') or {}).get('accuracy'),
                    timeout=summary.get('timeout_reached', False),
                )
                if preset.use_cache and cache_key:
                    self._last_cache_info = {'used': False, 'key': cache_key}
                    save_cached_training(cache_key, {
                        'model': model,
                        'X_encoded': X_encoded,
                        'y_encoded': y_encoded,
                        'eval_split': self.trepan.mlp_trainer.eval_split,
                        'mlp_trainer_state': serialize_mlp_trainer_state(self.trepan.mlp_trainer),
                        'metrics': summary.get('test_metrics'),
                        'optimization_method': summary.get('optimization_method'),
                        'n_features': n_orig_features,
                        'feature_space': 'original',
                        'feature_names': orig_feature_names,
                        'bundle': getattr(self.trepan.mlp_trainer, 'bundle', None),
                    })

            if model is not None and self.mlp_model is None:
                self.mlp_model = model

            if self.mlp_model is not None:
                self.mlp_model, self.mlp_original_diagnostic = (
                    self._diagnose_and_harden_mlp_original(self.mlp_model)
                )

            if has_ontology:
                progress("onto_gate", 38, "Validando OWL e ajustando fit/transform no treino...")
                self._prepare_ontology_train_test()
                dual_pipeline = self._uses_dual_data_pipeline()

            X_encoded_aug = None
            y_encoded_aug = None
            feature_names_aug = None

            if dual_pipeline:
                progress("onto", 40, "Preparando features enriquecidas...")
                X_encoded_aug, y_encoded_aug = self._encode_augmented_for_tree()
                feature_names_aug = self.feature_names_augmented
                orig_names = self._get_original_feature_names()

                self.X_encoded_aug = X_encoded_aug
                self.y_encoded_aug = y_encoded_aug

                progress("mlp_residual", 50, "Otimizando professor híbrido ontológico...")
                t_onto = perf.start_stage("Professor Híbrido Ontológico")
                residual_msg, residual_oracle, acceptance, selected_oracle = (
                    self.trepan.train_mlp_residual_onto(
                        X_encoded_aug,
                        y_encoded_aug,
                        self.mlp_model,
                        orig_names,
                        feature_names_aug,
                        dataset_name=dataset_name,
                        preset=preset,
                        cancel_fn=cancel_fn,
                        progress_fn=progress_fn,
                        acceptance_tolerance=preset.ontology_acceptance_tolerance,
                        top_k_onto_features=preset.top_k_onto_features,
                        X_train_aug=self._eval_split_augmented['X_train'],
                        X_test_aug=self._eval_split_augmented['X_test'],
                        X_train_raw=self._eval_split_augmented.get('X_train_raw'),
                        y_train=self._eval_split_augmented['y_train'],
                        y_test=self._eval_split_augmented['y_test'],
                        ontology_quality_report=self.ontology_quality_report,
                    )
                )
                if cancelled():
                    raise InterruptedError("Entrenamiento cancelado.")
                if residual_oracle is not None and acceptance is not None:
                    selected_indices = acceptance.get('selected_feature_indices')
                    selected_names = acceptance.get('selected_feature_names')
                    if selected_indices is not None and selected_names is not None:
                        from core.onto_feature_selector import apply_ontology_feature_selection
                        X_encoded_aug = apply_ontology_feature_selection(
                            X_encoded_aug, selected_indices
                        )
                        feature_names_aug = list(selected_names)
                        self.X_encoded_aug = X_encoded_aug
                        self.feature_names_augmented = feature_names_aug
                        if self.ontology_transformer is not None:
                            selected_set = set(feature_names_aug)
                            self.ontology_transformer.feature_specs_ = [
                                spec for spec in self.ontology_transformer.feature_specs_
                                if spec['name'] in selected_set
                            ]
                            self.ontology_transformer.output_features_ = list(feature_names_aug)
                            self.trepan.extractor.ontology_processor = self.ontology_transformer
                        print(
                            "[INFO] Seleção ontológica ajustada no treino e "
                            f"aplicada ao dataset: {acceptance.get('feature_selection')}"
                        )
                    self.mlp_model_residual = residual_oracle
                    self.mlp_model_onto = (
                        selected_oracle
                        if acceptance.get('accepted')
                        else None
                    )
                    self.mlp_model_reloaded = selected_oracle
                    self.selected_oracle = selected_oracle
                    self.selected_oracle_label = self.trepan.selected_oracle_label
                    self.ontology_acceptance = acceptance
                    self.mlp_trainer_reloaded = self.trepan.mlp_trainer_residual
                    aug_eval = getattr(self.trepan, 'augmented_eval_split', None)
                    if aug_eval is not None:
                        self._eval_split_augmented = dict(aug_eval)
                    if not acceptance.get('accepted'):
                        # O gate estrito já decidiu o fallback usando apenas
                        # desenvolvimento. Não reabrir essa decisão olhando o
                        # teste bloqueado: usar directamente a versão base.
                        self.selected_oracle = self.mlp_model
                        self.selected_oracle_label = "MLP Original"
                        self.reloaded_oracle_diagnostic = {
                            'selection_role': 'strict_gate_fallback',
                            'test_used_for_selection': False,
                            'degenerate': False,
                        }
                    else:
                        (
                            self.selected_oracle,
                            self.selected_oracle_label,
                            self.reloaded_oracle_diagnostic,
                        ) = self._ensure_viable_reloaded_oracle(
                            self.mlp_model,
                            self.selected_oracle,
                            self.selected_oracle_label,
                        )
                    if not acceptance.get('accepted') or (self.reloaded_oracle_diagnostic or {}).get('degenerate'):
                        self.mlp_model_onto = None
                        self.mlp_model_reloaded = self.selected_oracle
                    onto_summary = (
                        self.trepan.mlp_trainer_residual.last_optimization_summary or {}
                    )
                    perf.end_stage(
                        "Professor Híbrido Ontológico",
                        t_onto,
                        n_features=X_encoded_aug.shape[1] if X_encoded_aug is not None else None,
                        n_trials=onto_summary.get('n_trials'),
                        best_score=(onto_summary.get('test_metrics') or {}).get('accuracy'),
                        timeout=onto_summary.get('timeout_reached', False),
                    )
                else:
                    perf.end_stage(
                        "Professor Híbrido Ontológico",
                        t_onto,
                        extra={'error': residual_msg},
                    )
                    self.mlp_model_reloaded = self.mlp_model
                    self.selected_oracle = self.mlp_model
                    self.selected_oracle_label = "MLP Original"
                    if acceptance is not None:
                        self.ontology_acceptance = acceptance
                        self.trepan.ontology_acceptance = acceptance
                        print(
                            "[INFO] "
                            + str(acceptance.get(
                                'reason', 'ONTOLOGY_VALID_BUT_NO_PREDICTIVE_UTILITY'
                            ))
                        )

            if model is not None or self.mlp_model is not None:
                if model is not None:
                    self.mlp_model = model
                if X_encoded is not None:
                    self.X_encoded = X_encoded
                    self.y_encoded = y_encoded

                eval_split = getattr(self.trepan.mlp_trainer, 'eval_split', None)
                if eval_split is not None:
                    self._eval_split_original = dict(eval_split)
                    X_test_eval = eval_split['X_test']
                    y_test_eval = eval_split['y_test']
                    y_pred_eval = self.mlp_model.predict(X_test_eval)
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_test_eval, y_pred_eval)
                    opt_summary = getattr(
                        self.trepan.mlp_trainer, 'last_optimization_summary', None
                    ) or {}
                    opt_method = opt_summary.get('optimization_method', 'optimized')
                    opt_note = f" (método: {opt_method}, modo: {preset.display_name})"
                else:
                    y_pred = self.mlp_model.predict(X_encoded)
                    y_true_labels = y
                    y_pred_labels = self.trepan.label_encoder.inverse_transform(y_pred)
                    from sklearn.metrics import accuracy_score
                    accuracy = accuracy_score(y_true_labels, y_pred_labels)
                    opt_note = " (muestra completa)"
                
                meta = self.trepan.mlp_trainer.arff_meta
                if meta is None:
                    feature_names = self._get_original_feature_names()
                    class_names = list(self.trepan.label_encoder.classes_)
                    meta = {
                        'features': feature_names,
                        'classes': class_names,
                        'target': 'target',
                        'file_name': 'unknown',
                        'sample_count': len(X_mlp)
                    }
                    self.trepan.mlp_trainer.set_metadata(
                        'unknown', feature_names, 'target', class_names, len(X_mlp)
                    )
                else:
                    feature_names = meta.get('original_features') or meta['features']
                    class_names = list(self.trepan.label_encoder.classes_)

                self.feature_names_original = feature_names
                if feature_names_aug is None:
                    feature_names_aug = feature_names
                
                tree_rules_all = ""
                
                from core.c45_j48_tree import C45Tree
                from sklearn.model_selection import train_test_split

                progress("c45", 62, "Entrenando C4.5 nativo...")
                t_c45 = perf.start_stage("C4.5")

                if self._eval_split_original is not None:
                    X_train_dom = self._eval_split_original['X_train']
                    y_train_dom = self._eval_split_original['y_train']
                    X_eval = self._eval_split_original['X_test']
                    y_eval = self._eval_split_original['y_test']
                    self._dominance_train_indices = len(X_train_dom)
                else:
                    try:
                        X_train_dom, X_eval, y_train_dom, y_eval = train_test_split(
                            X_encoded, y_encoded, test_size=0.3, random_state=42, stratify=y_encoded
                        )
                    except ValueError:
                        X_train_dom, X_eval, y_train_dom, y_eval = train_test_split(
                            X_encoded, y_encoded, test_size=0.3, random_state=42
                        )
                    self._dominance_train_indices = len(X_train_dom)

                c45_tree_obj = C45Tree()
                c45_tree_obj.train_c45_tree(
                    X_train_dom, y_train_dom, feature_names, class_names,
                    ontology_mode=False,
                    feature_types=self._get_original_feature_types(),
                )
                self.c45_tree = c45_tree_obj.tree_model
                # O Reloaded consulta explicitamente as regiões em que o seu
                # oráculo e o C4.5 divergem; o baseline usa apenas features ARFF.
                self.trepan.extractor.c45_baseline = self.c45_tree
                perf.end_stage("C4.5", t_c45)
                tree_rules_all += (
                    f"\n🌳 C4.5-Nativo (datos originales, etiquetas reales):\n"
                    f"Entrenada con {len(X_train_dom)} muestras.\n"
                )

                progress("trepan_original", 72, "Extrayendo Trepan Original...")
                t_trep_o = perf.start_stage("Trepan Original")
                from core.trepan_original import TrepanOriginalExtractor
                original_extractor = TrepanOriginalExtractor()
                trepan_limits = self._trepan_training_limits(preset)

                # Modo Científico: selecciona a capacidade COMUM do TREPAN
                # exclusivamente por CV interna do treino. O mesmo orçamento
                # seleccionado é reutilizado pelo Reloaded; o teste externo não
                # participa desta escolha.
                self._trepan_scientific_tuning = None
                if getattr(preset, 'trepan_scientific_tuning', True):
                    from core.controlled_trepan_experiment import ControlledTrepanConfig
                    from core.trepan_scientific_tuning import (
                        ScientificTrepanSearchConfig, tune_scientific_trepan,
                    )
                    tune_X = (
                        self._eval_split_original['X_train']
                        if self._eval_split_original is not None else X_encoded
                    )
                    tune_y = (
                        self._eval_split_original['y_train']
                        if self._eval_split_original is not None else y_encoded
                    )
                    tune_X = np.asarray(tune_X, dtype=float)
                    tune_y = np.asarray(tune_y)
                    base_cfg = ControlledTrepanConfig(
                        max_nodes=int(trepan_limits['max_nodes']),
                        max_depth=int(trepan_limits['max_depth']),
                        min_samples_leaf=int(trepan_limits['min_samples_leaf']),
                        min_sample=max(len(tune_X), min(1000, max(120, len(tune_X) * 3))),
                        max_n=int(getattr(preset, 'canonical_m_of_n_max_n', 3)),
                        beam_width=2,
                        max_features_per_node=min(tune_X.shape[1], max(12, min(32, tune_X.shape[1]))),
                        max_queries=int(trepan_limits['max_queries']),
                        random_state=42,
                        semantic_gain_strength=float(getattr(preset, 'semantic_gain_strength', 1.0)),
                        semantic_group_strength=float(getattr(preset, 'semantic_group_strength', 0.15)),
                        semantic_candidate_budget=int(getattr(preset, 'semantic_candidate_budget', 24)),
                        semantic_relation_threshold=float(getattr(preset, 'semantic_relation_threshold', 0.35)),
                        semantic_active_query_fraction=float(getattr(preset, 'semantic_active_query_fraction', 0.65)),
                        semantic_active_pool_multiplier=int(getattr(preset, 'semantic_active_pool_multiplier', 4)),
                        error_focused_refinement=bool(getattr(preset, 'error_focused_refinement', True)),
                        error_focus_min_disagreement=float(getattr(preset, 'error_focus_min_disagreement', 0.05)),
                        error_focus_strength=float(getattr(preset, 'error_focus_strength', 1.0)),
                        error_focus_semantic_weight=float(getattr(preset, 'error_focus_semantic_weight', 0.50)),
                        error_focus_uncertainty_weight=float(getattr(preset, 'error_focus_uncertainty_weight', 0.35)),
                        error_focus_min_local_fidelity_gain=float(getattr(preset, 'error_focus_min_local_fidelity_gain', 0.002)),
                        error_focus_min_real_fidelity_gain=float(getattr(preset, 'error_focus_min_real_fidelity_gain', 0.0)),
                        error_focus_anchor_k=int(getattr(preset, 'error_focus_anchor_k', 24)),
                        error_focus_top_k=int(getattr(preset, 'error_focus_top_k', 3)),
                        error_focus_min_regions=int(getattr(preset, 'error_focus_min_regions', 1)),
                        mirror_when_no_semantic_effect=bool(getattr(preset, 'mirror_when_no_semantic_effect', True)),
                    )
                    self._trepan_scientific_tuning = tune_scientific_trepan(
                        tune_X, tune_y, oracle=self.mlp_model, feature_names=feature_names,
                        base_config=base_cfg,
                        search=ScientificTrepanSearchConfig(
                            cv_folds=int(getattr(preset, 'trepan_tuning_cv_folds', 3)),
                            max_capacity_candidates=int(getattr(preset, 'trepan_tuning_capacity_candidates', 6)),
                            max_semantic_candidates=1,
                            fidelity_target=float(getattr(preset, 'trepan_fidelity_tuning_target', 0.95)),
                        ),
                    )
                    common = self._trepan_scientific_tuning['common_capacity']
                    trepan_limits.update({
                        'max_nodes': int(common['max_nodes']),
                        'max_depth': int(common['max_depth']),
                        'max_queries': int(common['max_queries']),
                        'min_sample': int(common['min_sample']),
                        'min_samples_leaf': int(common['min_samples_leaf']),
                        'max_n': int(common['max_n']),
                        'beam_width': int(common['beam_width']),
                    })
                    # Propaga a MESMA capacidade ao extractor Reloaded.
                    self.trepan.extractor._training_limits.update({
                        'canonical_max_nodes': int(common['max_nodes']),
                        'canonical_max_depth': int(common['max_depth']),
                        'historical_max_queries': int(common['max_queries']),
                        'historical_min_sample': int(common['min_sample']),
                        'historical_min_samples_leaf': int(common['min_samples_leaf']),
                        'canonical_m_of_n_max_n': int(common['max_n']),
                        'historical_beam_width': int(common['beam_width']),
                        'historical_max_features_per_node': int(common['max_features_per_node']),
                        'trepan_scientific_tuning': True,
                        'trepan_tuning_cv_folds': int(getattr(preset, 'trepan_tuning_cv_folds', 3)),
                        'trepan_fidelity_tuning_target': float(getattr(preset, 'trepan_fidelity_tuning_target', 0.95)),
                        'semantic_tuning_candidates': int(getattr(preset, 'semantic_tuning_candidates', 6)),
                        'semantic_gain_strength': float(getattr(preset, 'semantic_gain_strength', 1.0)),
                        'semantic_group_strength': float(getattr(preset, 'semantic_group_strength', 0.15)),
                        'semantic_candidate_budget': int(getattr(preset, 'semantic_candidate_budget', 24)),
                        'semantic_relation_threshold': float(getattr(preset, 'semantic_relation_threshold', 0.35)),
                        'semantic_active_query_fraction': float(getattr(preset, 'semantic_active_query_fraction', 0.65)),
                        'semantic_active_pool_multiplier': int(getattr(preset, 'semantic_active_pool_multiplier', 4)),
                        'error_focused_refinement': bool(getattr(preset, 'error_focused_refinement', True)),
                        'error_focus_min_disagreement': float(getattr(preset, 'error_focus_min_disagreement', 0.05)),
                        'error_focus_strength': float(getattr(preset, 'error_focus_strength', 1.0)),
                        'error_focus_semantic_weight': float(getattr(preset, 'error_focus_semantic_weight', 0.50)),
                        'error_focus_uncertainty_weight': float(getattr(preset, 'error_focus_uncertainty_weight', 0.35)),
                        'error_focus_min_local_fidelity_gain': float(getattr(preset, 'error_focus_min_local_fidelity_gain', 0.002)),
                        'error_focus_min_real_fidelity_gain': float(getattr(preset, 'error_focus_min_real_fidelity_gain', 0.0)),
                        'error_focus_anchor_k': int(getattr(preset, 'error_focus_anchor_k', 24)),
                        'error_focus_top_k': int(getattr(preset, 'error_focus_top_k', 3)),
                        'error_focus_min_regions': int(getattr(preset, 'error_focus_min_regions', 1)),
                        'mirror_when_no_semantic_effect': bool(getattr(preset, 'mirror_when_no_semantic_effect', True)),
                    })

                trepan_train_kwargs = dict(
                    sample_size=preset.trepan_sample_size,
                    feature_names=feature_names,
                    class_names=class_names,
                    training_limits=trepan_limits,
                )
                if self._eval_split_original is not None:
                    split = self._eval_split_original
                    trepan_train_kwargs.update(
                        X_train=split['X_train'],
                        X_test=split['X_test'],
                        y_train=split['y_train'],
                        y_test=split['y_test'],
                    )
                tree_rules_original = original_extractor.extract_tree(
                    self.mlp_model, X_encoded, y_encoded,
                    **trepan_train_kwargs,
                )
                self.trepan_original_tree = original_extractor.explainer_tree
                self.trepan_original_audit = dict(
                    getattr(original_extractor, 'last_audit', {}) or {}
                )
                perf.end_stage(
                    "Trepan Original",
                    t_trep_o,
                    best_score=self.trepan_original_audit.get('trepan_fidelity'),
                )
                tree_rules_all += f"\n🌳 Trepan-Original (oráculo MLP original):\n{tree_rules_original}\n"
                
                progress("trepan_reloaded", 85, "Extrayendo Trepan Reloaded...")
                t_trep_r = perf.start_stage("Trepan Reloaded")
                reloaded_label = "Trepan-Reloaded (= Trepan-Original)"
                tree_rules_reloaded = tree_rules_original

                if not has_ontology and not dual_pipeline:
                    self.trepan_reloaded_audit = self.trepan.mirror_original_tree(
                        self.trepan_original_tree,
                        feature_names=feature_names,
                        original_audit=self.trepan_original_audit,
                    )
                    self.trepan_reloaded_tree = self.trepan_original_tree
                    self.trepan_reloaded_feature_names = list(feature_names)
                else:
                    from core.trepan_reloaded_context import log_trepan_reloaded_context

                    reloaded_ctx = self._get_trepan_reloaded_context(
                        feature_names, feature_names_aug
                    )
                    self._trepan_reloaded_context = reloaded_ctx
                    log_trepan_reloaded_context(reloaded_ctx)

                    reloaded_train_kwargs = dict(
                        sample_size=preset.trepan_sample_size,
                        original_feature_names=feature_names,
                        mlp_model_onto=reloaded_ctx.get('mlp_model_onto'),
                        X_train=reloaded_ctx['X_train'],
                        X_test=reloaded_ctx['X_test'],
                        y_train=reloaded_ctx['y_train'],
                        y_test=reloaded_ctx['y_test'],
                        reloaded_context=reloaded_ctx,
                    )

                    matrix_for_extract = reloaded_ctx['X_train']
                    names_for_extract = reloaded_ctx['feature_names']
                    y_for_extract = reloaded_ctx['y_train']

                    tree_rules_reloaded = self.trepan.extractor.extract_tree_with_ontology(
                        self.mlp_model,
                        matrix_for_extract,
                        y_for_extract,
                        names_for_extract,
                        class_names,
                        **reloaded_train_kwargs,
                    )
                    oracle_note = reloaded_ctx.get('oracle_name', 'MLP Original')
                    space_note = reloaded_ctx.get('feature_space', 'original')
                    reloaded_label = (
                        f"Trepan-Reloaded (espacio {space_note}, oráculo {oracle_note})"
                    )
                    self.trepan_reloaded_tree = self.trepan.extractor.explainer_tree
                    self.trepan_reloaded_feature_names = list(
                        reloaded_ctx.get('feature_names') or names_for_extract
                    )
                    self.trepan_reloaded_audit = dict(
                        getattr(self.trepan.extractor, 'last_audit', {}) or {}
                    )
                perf.end_stage(
                    "Trepan Reloaded",
                    t_trep_r,
                    best_score=(getattr(self, 'trepan_reloaded_audit', None) or {}).get(
                        'trepan_fidelity'
                    ),
                )
                tree_rules_all += f"\n🧠 {reloaded_label}:\n{tree_rules_reloaded}\n"

                if has_ontology or dual_pipeline:
                    tree_rules_all += (
                        "\n✅ Trepan-Reloaded entrenado.\n"
                        "💡 Compare métricas en la pestaña dedicada.\n"
                    )

                cache = getattr(self.trepan.extractor, '_training_cache', None) or {}
                self.trepan_reloaded_feature_names = (
                    cache.get('feature_names')
                    or getattr(self.trepan.extractor, '_matrix_feature_names', None)
                    or feature_names_aug
                    or feature_names
                )

                mapping_info = ""
                if has_ontology and hasattr(self.trepan.extractor, 'get_mapping_statistics'):
                    try:
                        stats = self.trepan.extractor.get_mapping_statistics()
                        mapped_pct = (stats['features']['mapped'] / stats['features']['total'] * 100) if stats['features']['total'] > 0 else 0
                        owl_meta = dict(getattr(self, 'loaded_ontology_metadata', {}) or {})
                        rel_stats = dict(stats.get('relationships') or {})
                        mapping_info = f"""
📊 Estatísticas de Mapeamento Ontológico:
   Features mapeadas: {stats['features']['mapped']}/{stats['features']['total']} ({mapped_pct:.1f}%)
   Fonte do mapeamento: {stats['features'].get('source', 'n/a')}
   Relações OWL disponíveis: {rel_stats.get('total_relationships', 0)}
   Features com grupos semânticos: {rel_stats.get('features_with_groups', 0)}
   OWL activo: {owl_meta.get('file_name', 'n/a')}
   VersionInfo: {owl_meta.get('version_info', 'n/a')}
   SHA-256: {owl_meta.get('sha256', 'n/a')}
"""
                    except Exception as e:
                        print(f"Error ao obter estatísticas de mapeamento: {e}")

                from sklearn.metrics import (
                    accuracy_score,
                    precision_score,
                    recall_score,
                    f1_score,
                    balanced_accuracy_score,
                )
                c45_metrics = None
                if self.c45_tree is not None:
                    y_pred_c45 = self.c45_tree.predict(X_eval)
                    precision_macro_c45 = float(
                        precision_score(y_eval, y_pred_c45, average='macro', zero_division=0)
                    )
                    recall_macro_c45 = float(
                        recall_score(y_eval, y_pred_c45, average='macro', zero_division=0)
                    )
                    macro_f1_c45 = float(
                        f1_score(y_eval, y_pred_c45, average='macro', zero_division=0)
                    )
                    c45_metrics = {
                        'accuracy': float(accuracy_score(y_eval, y_pred_c45)),
                        # aliases legados apontam agora para as métricas macro,
                        # coerentes com o gate científico e com a interface.
                        'precision': precision_macro_c45,
                        'precision_macro': precision_macro_c45,
                        'precision_weighted': float(
                            precision_score(y_eval, y_pred_c45, average='weighted', zero_division=0)
                        ),
                        'recall': recall_macro_c45,
                        'recall_macro': recall_macro_c45,
                        'f1': macro_f1_c45,
                        'macro_f1': macro_f1_c45,
                        'balanced_accuracy': float(balanced_accuracy_score(y_eval, y_pred_c45)),
                        'unique_predictions': int(len(np.unique(y_pred_c45))),
                        'metric_semantics': {
                            'precision': 'precision_macro_alias',
                            'precision_weighted': 'audit_only',
                        },
                    }
                self._last_c45_metrics = c45_metrics

                progress("audit", 95, "Calculando métricas y auditoría...")
                try:
                    from core.pipeline_audit import log_pipeline_training_audit
                    mlp_orig_summary = getattr(
                        self.trepan.mlp_trainer, 'last_optimization_summary', None
                    )
                    mlp_onto_summary = None
                    if self.mlp_model_residual is not None:
                        mlp_onto_summary = getattr(
                            self.trepan.mlp_trainer_residual, 'last_optimization_summary', None
                        )
                    log_pipeline_training_audit(
                        dataset_name=meta.get('file_name', 'unknown'),
                        mlp_original_model=self.mlp_model,
                        mlp_onto_model=self.mlp_model_residual,
                        mlp_original_summary=mlp_orig_summary,
                        mlp_onto_summary=mlp_onto_summary,
                        trepan_original_audit=getattr(self, 'trepan_original_audit', None),
                        trepan_reloaded_audit=getattr(self, 'trepan_reloaded_audit', None),
                        c45_metrics=c45_metrics,
                        ontology_acceptance=self.ontology_acceptance,
                        selected_oracle_label=self.selected_oracle_label,
                    )
                except Exception as audit_exc:
                    print(f"[WARN] Auditoria de treino falhou: {audit_exc}")

                perf_path = perf.save()
                perf_summary = perf.summary()
                
                cache_note = " (caché utilizada)" if perf.used_cache else ""
                timeout_note = " ⚠️ Timeout parcial." if perf.stopped_by_timeout else ""

                ontology_metrics_info = ""
                if has_ontology or self.ontology_quality_report or self.ontology_acceptance:
                    from gui.ontology_status_presenter import build_ontology_status_text
                    ontology_metrics_info = "\n" + build_ontology_status_text(
                        self.ontology_quality_report,
                        self.ontology_acceptance,
                        selected_oracle_label=(self.selected_oracle_label or 'MLP Original'),
                    ) + "\n"
                    if self.ontology_acceptance is not None:
                        surrogate = self.ontology_acceptance.get('surrogate_gate_accepted')
                        surrogate_label = (
                            'ACEITE' if surrogate is True
                            else 'REJEITADO' if surrogate is False
                            else 'PENDENTE / não avaliado nesta etapa'
                        )
                        ontology_metrics_info += f"   Gate da árvore substituta: {surrogate_label}\n"

                result_text = f"""
✅ ¡Modelo entrenado con éxito!

⚙️ Modo: {preset.display_name}{cache_note}{timeout_note}

📊 Rendimiento MLP (prueba): {accuracy:.1%}{opt_note}
{ontology_metrics_info}
{tree_rules_all}
{mapping_info}
📈 Registro de rendimiento: {perf_path.name}

Claridad alcanzada: el MLP ya tiene un árbol sustituto.
Próximos pasos en Herramientas avanzadas:
• Visualizar Árbol — recorrer las reglas
• Comparar Métricas — fidelidad frente al oráculo
• Explicaciones Naturales — leer sin jerga técnica
                """
                
                progress("done", 100, "Entrenamiento completado.")
                return {
                    'success': True,
                    'result_text': result_text,
                    'performance_summary': perf_summary,
                }
                
            return {'success': False, 'result_text': f"❌ Error en el entrenamiento: {message}"}
                
        except InterruptedError:
            perf.mark_cancelled()
            perf.save()
            raise
        except Exception as e:
            raise

    def generate_explanation(self):
        if not self._has_loaded_data():
            QMessageBox.warning(self, "Advertencia", "¡Cargue datos primero!")
            return
        if self._warn_if_busy("generar una explicación"):
            return

        try:
            self.show_progress("Generando explicaciones...")
            
            X, y = self._get_original_xy()
            
            if not self.mlp_model:
                message, model, X_encoded, y_encoded = self.trepan.train_mlp(X, y)
                if model:
                    self.mlp_model = model
                else:
                    self.show_results(f"❌ Error al preparar análisis: {message}")
                    return
            else:
                X_encoded = self.trepan.mlp_trainer._encode_features(X)
                y_encoded = self.trepan.label_encoder.transform(y)
            
            y_pred = self.mlp_model.predict(X_encoded)
            y_true_labels = y
            y_pred_labels = self.trepan.label_encoder.inverse_transform(y_pred)
            
            from sklearn.metrics import accuracy_score
            accuracy = accuracy_score(y_true_labels, y_pred_labels)
            
            friendly_explanation = self.generate_user_friendly_explanation(y_true_labels, y_pred_labels, accuracy)
            
            meta = self.trepan.mlp_trainer.arff_meta
            if meta is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
                class_names = list(self.trepan.label_encoder.classes_)
            else:
                feature_names = meta.get('original_features') or meta['features']
                class_names = list(self.trepan.label_encoder.classes_)
            
            tree_rules = ""
            if self.loaded_ontology is not None and self.trepan_reloaded_tree is not None:
                if hasattr(self.trepan.extractor, 'generate_semantic_explanations'):
                    tree_rules = self.trepan.extractor.generate_semantic_explanations(
                        self.trepan_reloaded_feature_names
                        or self.feature_names_augmented
                        or feature_names,
                        class_names,
                    )
                else:
                    tree_rules = (
                        "Árbol Trepan-Reloaded ya entrenado — "
                        "use Visualizar Árbol para explorar las reglas."
                    )
            else:
                from core.trepan_original import TrepanOriginalExtractor
                explainer = TrepanOriginalExtractor()
                tree_rules = explainer.extract_tree(
                    self.mlp_model, X_encoded, y_encoded,
                    sample_size=2000,
                    feature_names=feature_names,
                    class_names=class_names,
                )
            
            result_text = f"""
Explicación lista — la red ya habla en reglas

{friendly_explanation}

Reglas del árbol sustituto:
{tree_rules}

Use Visualizar Árbol para recorrer cada decisión,
o Comparar Métricas para medir cuánto el árbol copia al MLP.
            """

            self.show_results(result_text)
            self.explanation_generated = True
            self._unlock_clarity(
                "modelo",
                status="Explicación lista. Las reglas del árbol ya están en Resultados.",
                nudge_advanced=True,
            )
            self._unlock_clarity("arbol")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar explicaciones: {str(e)}")
            
    def generate_user_friendly_explanation(self, y_true, y_pred, accuracy):
        import numpy as np
        from sklearn.metrics import precision_recall_fscore_support
        
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
        classes = np.unique(y_true)
        
        explanation = f"🎯 El modelo acertó {accuracy:.1%} de las predicciones!\n\n"
        
        if accuracy > 0.9:
            explanation += "🌟 ¡Excelente! Su modelo funciona muy bien.\n"
        elif accuracy > 0.7:
            explanation += "👍 ¡Buen trabajo! El modelo tiene un rendimiento sólido.\n"
        else:
            explanation += "💪 Hay espacio para mejoras. Analicemos los detalles.\n"
        
        explanation += f"\n📊 Análisis por categoría:\n"
        
        for i, cls in enumerate(classes):
            prec = precision[i]
            rec = recall[i]
            f1_score = f1[i]
            
            explanation += f"\n🏷️ Categoría '{cls}':\n"
            
            if prec > 0.8:
                explanation += f"   ✅ Cuando el modelo dice '{cls}', acierta {prec:.0%} de las veces\n"
            else:
                explanation += f"   ⚠️ Cuando el modelo dice '{cls}', acierta {prec:.0%} de las veces\n"
            
            if rec > 0.8:
                explanation += f"   ✅ El modelo encuentra {rec:.0%} de los casos reales de '{cls}'\n"
            else:
                explanation += f"   ⚠️ El modelo encuentra {rec:.0%} de los casos reales de '{cls}'\n"
        
        explanation += f"\n💡 En resumen: "
        if accuracy > 0.9:
            explanation += "¡Su modelo es muy confiable para tomar decisiones!"
        elif accuracy > 0.7:
            explanation += "Su modelo es útil, pero puede refinarse con más datos."
        else:
            explanation += "Considere recopilar más datos o ajustar el modelo."
        
        return explanation
        
    def visualize_tree(self):
        if not self.mlp_model:
            QMessageBox.warning(self, "Advertencia", "¡Entrene un modelo primero!")
            return
        if self._warn_if_busy("visualizar el árbol"):
            return

        try:
            meta = self.trepan.mlp_trainer.arff_meta
            if meta is None:
                feature_names = [f"feature_{i}" for i in range(len(self.trepan.feature_encoders))]
                class_names = list(self.trepan.label_encoder.classes_)
                meta = {
                    'features': feature_names,
                    'classes': class_names,
                    'target': 'target',
                    'file_name': 'unknown',
                    'sample_count': 0
                }
            else:
                feature_names = meta.get('original_features') or meta['features']
                class_names = list(self.trepan.label_encoder.classes_)
            
            feature_names_reloaded = (
                self.trepan_reloaded_feature_names
                or self.feature_names_augmented
                or getattr(self.trepan.extractor, '_matrix_feature_names', None)
            )
            if not feature_names_reloaded:
                cache = getattr(self.trepan.extractor, '_training_cache', None) or {}
                feature_names_reloaded = cache.get('feature_names') or feature_names

            dataset_name = meta.get('file_name', 'dataset')
            viz = TreeVisualizationWidget(
                tree_model=self.trepan.extractor.explainer_tree,
                feature_names=feature_names,
                feature_names_reloaded=feature_names_reloaded,
                class_names=class_names,
                trepan_original_tree=self.trepan_original_tree,
                trepan_reloaded_tree=self.trepan_reloaded_tree,
                c45_tree=self.c45_tree,
                trepan_original_improved_tree=self.trepan_improved_tree,
                trepan_reloaded_improved_tree=self.trepan_reloaded_improved_tree,
                dataset_name=dataset_name,
            )
            viz._extractor_tree_names = (
                getattr(self.trepan.extractor, '_training_cache', {}) or {}
            ).get('feature_names')
            self.tree_widget = viz
            
            layout = self.visualization_tab.layout()
            if layout:
                for i in reversed(range(layout.count())):
                    layout.itemAt(i).widget().setParent(None)
            
            layout.addWidget(self.tree_widget)

            # Abrir siempre la pestaña Visualización (no pasar por show_results,
            # que vuelve a Resultados).
            self.content_tabs.setCurrentWidget(self.visualization_tab)
            self._set_status("Árbol actualizado — pestaña Visualización.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al visualizar árbol: {str(e)}")
            
    def compare_metrics(self):
        if not self._has_loaded_data():
            QMessageBox.warning(self, "Advertencia", "¡Cargue datos primero!")
            return
        if self._warn_if_busy("comparar métricas"):
            return

        X, y = self._get_original_xy()
        X = np.array(X)
        y = np.array(y)

        try:
            X_for_validation = np.array(X, dtype=float)
        except (ValueError, TypeError):
            X_for_validation = X

        validation_issues = self._validate_data_for_comparison(X_for_validation, y)
        if validation_issues:
            msg = "Advertencias sobre los datos:\n\n" + "\n".join(
                f"• {issue}" for issue in validation_issues
            )
            msg += "\n\n¿Desea continuar de todos modos?"
            reply = QMessageBox.question(
                self,
                "Advertencias de Validación",
                msg,
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        self._pending_failure_detail = None
        if self.audit is not None:
            self.audit.sm.begin()
            self.audit.apply_buttons()
        dlg = QProgressDialog(progress_text("compare"), tr("progress.cancel"), 0, 0, self)
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setWindowTitle(tr("progress.title.metrics"))
        dlg.setAutoClose(False)
        dlg.setMinimumDuration(0)
        self._metrics_progress_dialog = dlg

        worker = MetricsWorker(self, X, y)
        self._metrics_worker = worker
        worker.progress.connect(self._on_metrics_progress)
        worker.finished_ok.connect(lambda r: self._on_metrics_finished(r, dlg))
        worker.failed_detail.connect(self._store_failure_detail)
        worker.failed.connect(lambda e: self._on_metrics_failed(e, dlg))
        dlg.canceled.connect(worker.request_cancel)
        dlg.show()
        worker.start()

    def _on_metrics_progress(self, stage, pct, message):
        # Keep a local ref: setValue() can re-enter the event loop and clear
        # self._metrics_progress_dialog before setLabelText runs.
        dlg = self._metrics_progress_dialog
        if dlg is None:
            return
        if self._metrics_progress_dialog is dlg:
            dlg.setLabelText(progress_text(stage))

    def _disconnect_metrics_worker(self):
        worker = self._metrics_worker
        if worker is not None:
            for signal, slot in (
                (worker.progress, self._on_metrics_progress),
            ):
                try:
                    signal.disconnect(slot)
                except TypeError:
                    pass
        self._metrics_worker = None
        self._metrics_progress_dialog = None

    def _on_metrics_finished(self, result, dlg):
        self._disconnect_metrics_worker()
        dlg.close()

        if result.get("c45_tree") is not None:
            self.c45_tree = result["c45_tree"]

        comparison_results = result.get("comparison_results")
        if (
            hasattr(self, "metrics_widget")
            and self.metrics_widget is not None
            and comparison_results is not None
        ):
            self.metrics_widget.update_comparison_results(comparison_results)

        self.content_tabs.setCurrentWidget(self.metrics_tab)
        if result.get("result_text"):
            self.show_results(result["result_text"], switch_tab=False)
        if self.audit is not None:
            self.audit.on_metrics_finished()
        self._set_status("Comparación completada — pestaña Métricas.")

    def _on_metrics_failed(self, error_msg, dlg):
        self._disconnect_metrics_worker()
        dlg.close()
        details, self._pending_failure_detail = self._pending_failure_detail, None
        if "cancelad" in error_msg.lower():
            self._set_status(error_msg)
            self.show_results(f"⚠ {error_msg}", switch_tab=False)
            if self.audit is not None:
                self.audit.sm.end()
                self.audit.apply_buttons()
        elif self.audit is not None:
            msg = self.audit.on_failure(error_msg, what_key="error.comparison", where_key="error.metrics_where",
                                        action_key="error.action.comparison", details=details)
            self._show_structured_error(msg)
        else:
            QMessageBox.critical(
                self,
                "Error",
                f"Error en la comparación: {error_msg}\n\n"
                "Verifique los datos y que el modelo esté entrenado.",
            )

    def _execute_metrics_comparison(self, X, y, progress_fn=None, cancel_fn=None):
        """Heavy metrics work for MetricsWorker (no UI calls)."""

        def progress(stage, pct, message):
            if cancel_fn and cancel_fn():
                raise InterruptedError("Comparación cancelada por el usuario.")
            if progress_fn:
                progress_fn(stage, pct, message)

        from sklearn.model_selection import train_test_split

        progress("load", 10, "Cargando datos...")
        progress("split", 20, "Preparando división de evaluación...")

        test_row_ids = None
        if self._eval_split_original is not None:
            progress("split", 40, "Usando división de evaluación del entrenamiento MLP...")
            split = self._eval_split_original
            X_train_encoded = split["X_train"]
            X_test_encoded = split["X_test"]
            y_train_encoded = split["y_train"]
            y_test_encoded = split["y_test"]
            test_row_ids = split.get("test_indices")
        elif self.X_encoded is not None and self.y_encoded is not None:
            progress("split", 40, "Usando datos codificados del entrenamiento...")
            all_indices = np.arange(len(self.y_encoded))
            try:
                train_ids, test_ids = train_test_split(
                    all_indices, test_size=0.3, random_state=42,
                    stratify=self.y_encoded,
                )
            except ValueError:
                train_ids, test_ids = train_test_split(
                    all_indices, test_size=0.3, random_state=42,
                )
            train_ids = np.asarray(train_ids, dtype=int)
            test_ids = np.asarray(test_ids, dtype=int)
            X_train_encoded = np.asarray(self.X_encoded)[train_ids]
            X_test_encoded = np.asarray(self.X_encoded)[test_ids]
            y_train_encoded = np.asarray(self.y_encoded)[train_ids]
            y_test_encoded = np.asarray(self.y_encoded)[test_ids]
            test_row_ids = test_ids
        else:
            all_indices = np.arange(len(y))
            try:
                train_ids, test_ids = train_test_split(
                    all_indices, test_size=0.3, random_state=42, stratify=y
                )
            except ValueError:
                train_ids, test_ids = train_test_split(
                    all_indices, test_size=0.3, random_state=42
                )
            train_ids = np.asarray(train_ids, dtype=int)
            test_ids = np.asarray(test_ids, dtype=int)
            X_arr_raw = np.asarray(X, dtype=object)
            y_arr_raw = np.asarray(y)
            X_train, X_test = X_arr_raw[train_ids], X_arr_raw[test_ids]
            y_train, y_test = y_arr_raw[train_ids], y_arr_raw[test_ids]
            test_row_ids = test_ids

            progress("encode", 40, "Codificando datos...")
            X_train_encoded = self.trepan.mlp_trainer._encode_features(X_train)
            X_test_encoded = self.trepan.mlp_trainer._encode_features(X_test)
            y_train_encoded = self.trepan.label_encoder.transform(y_train)
            y_test_encoded = self.trepan.label_encoder.transform(y_test)

        dual_pipeline = (
            self._uses_dual_data_pipeline() and self.X_encoded_aug is not None
        )
        if dual_pipeline:
            if self._eval_split_augmented is not None:
                split_aug = self._eval_split_augmented
                # kept for parity with prior dual-pipeline prep
                _ = split_aug["X_train"]
                _ = split_aug["X_test"]
            else:
                try:
                    train_test_split(
                        self.X_encoded_aug,
                        self.y_encoded_aug,
                        test_size=0.3,
                        random_state=42,
                        stratify=self.y_encoded_aug,
                    )
                except ValueError:
                    train_test_split(
                        self.X_encoded_aug,
                        self.y_encoded_aug,
                        test_size=0.3,
                        random_state=42,
                    )

        meta = self.trepan.mlp_trainer.arff_meta
        if meta is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            class_names = list(self.trepan.label_encoder.classes_)
            meta = {
                "features": feature_names,
                "classes": class_names,
                "target": "target",
                "file_name": "unknown",
                "sample_count": len(X),
            }
        else:
            feature_names = meta.get("original_features") or meta["features"]
            class_names = list(self.trepan.label_encoder.classes_)

        feature_names_aug = self.feature_names_augmented or feature_names

        progress("compare", 60, "Ejecutando comparación de modelos...")
        self.metrics_comparator.clear_cache()

        try:
            from core.pipeline_audit import log_final_model_audit

            log_final_model_audit(
                mlp_diagnostic=getattr(self, "mlp_original_diagnostic", None),
                mlp_model=self.mlp_model,
                c45_metrics=getattr(self, "_last_c45_metrics", None),
                trepan_original_audit=getattr(self, "trepan_original_audit", None),
                trepan_reloaded_audit=getattr(self, "trepan_reloaded_audit", None),
                trepan_original_tree=self.trepan_original_tree,
                trepan_reloaded_tree=self.trepan_reloaded_tree,
                ontology_acceptance=self.ontology_acceptance,
                selected_oracle_label=self.selected_oracle_label,
                reloaded_oracle_diagnostic=getattr(
                    self, "reloaded_oracle_diagnostic", None
                ),
            )
        except Exception as audit_exc:
            print(f"[WARN] FINAL MODEL AUDIT falló: {audit_exc}")

        has_ontology = self.loaded_ontology is not None
        compare_kwargs = dict(
            mlp_model=self.mlp_model,
            trepan_original_tree=self.trepan_original_tree,
            trepan_reloaded_tree=self.trepan_reloaded_tree,
            X_test=X_test_encoded,
            y_test=y_test_encoded,
            X_train=X_train_encoded,
            y_train=y_train_encoded,
            feature_names=feature_names,
            class_names=class_names,
            ontology_active=has_ontology,
            feature_types=self._get_original_feature_types(),
            test_row_ids=test_row_ids,
            evaluation_seed=42,
            repeat_count=1,
            preprocessing_id=(
                "biuri_gui_original_encoder_plus_training_fitted_ontology_transform"
                if has_ontology else "biuri_gui_original_encoder"
            ),
        )

        if has_ontology and self.trepan_reloaded_tree is not None:
            ext = self.trepan.extractor
            reloaded_ctx = getattr(self, "_trepan_reloaded_context", None)
            if reloaded_ctx is None:
                reloaded_ctx = self._get_trepan_reloaded_context(
                    feature_names, feature_names_aug
                )
            reloaded_names = list(
                reloaded_ctx.get("feature_names")
                or self.trepan_reloaded_feature_names
                or feature_names
            )

            def _reload_tree_matrix(tree, X_mat):
                return ext._prepare_tree_matrix(tree, X_mat, reloaded_names)

            ontology_feature_metadata = None
            processor = getattr(ext, "ontology_processor", None)
            feature_audit = getattr(processor, "feature_audit_", None) if processor is not None else None
            if feature_audit:
                ontology_feature_metadata = {
                    str(item.get("feature")): dict(item)
                    for item in feature_audit
                    if isinstance(item, dict) and item.get("feature")
                }
            compare_kwargs.update(
                feature_names_reloaded=reloaded_names,
                original_feature_names=feature_names,
                reload_test_transform=_reload_tree_matrix,
                X_test_reloaded=reloaded_ctx["X_test"],
                X_train_reloaded=reloaded_ctx["X_train"],
                mlp_model_reloaded=reloaded_ctx["oracle"],
                ontology_feature_metadata=ontology_feature_metadata,
            )
        elif dual_pipeline:
            reloaded_ctx = getattr(self, "_trepan_reloaded_context", None)
            if reloaded_ctx is not None:
                compare_kwargs.update(
                    X_test_reloaded=reloaded_ctx["X_test"],
                    X_train_reloaded=reloaded_ctx["X_train"],
                    mlp_model_reloaded=reloaded_ctx["oracle"],
                )
            elif self.mlp_model_reloaded is not None:
                compare_kwargs["mlp_model_reloaded"] = self.mlp_model_reloaded
        elif has_ontology and self.augmented_data is not None:
            X_enc_aug_full, y_enc_aug_full = self._encode_augmented_for_tree()
            if X_enc_aug_full is not None:
                try:
                    X_tr_aug, X_te_aug, _, _ = train_test_split(
                        X_enc_aug_full,
                        y_enc_aug_full,
                        test_size=0.3,
                        random_state=42,
                        stratify=y_enc_aug_full,
                    )
                except ValueError:
                    X_tr_aug, X_te_aug, _, _ = train_test_split(
                        X_enc_aug_full,
                        y_enc_aug_full,
                        test_size=0.3,
                        random_state=42,
                    )
                compare_kwargs.update(
                    X_test_reloaded=X_te_aug,
                    X_train_reloaded=X_tr_aug,
                )

        if cancel_fn and cancel_fn():
            raise InterruptedError("Comparación cancelada por el usuario.")

        comparison_results = self.metrics_comparator.compare_all_models(**compare_kwargs)

        try:
            from core.mlp_optimizer import save_comparison_metrics_for_article

            opt_runs = []
            if hasattr(self.trepan.mlp_trainer, "optimization_results"):
                opt_runs.extend(self.trepan.mlp_trainer.optimization_results or [])
            if hasattr(self.trepan.mlp_trainer_residual, "optimization_results"):
                opt_runs.extend(
                    self.trepan.mlp_trainer_residual.optimization_results or []
                )
            ds_name = meta.get("file_name", "unknown") if meta else "unknown"
            save_comparison_metrics_for_article(
                comparison_results,
                dataset_name=ds_name,
                mlp_optimization_runs=opt_runs,
            )
        except Exception as export_exc:
            print(f"[WARN] Exportación métricas artículo falló: {export_exc}")

        try:
            from core.pipeline_audit import log_pipeline_comparison_audit

            fid_block = (comparison_results.get("fidelity") or {}).get(
                "trepan_reloaded"
            ) or {}
            fid_ref = fid_block.get("fidelity_reference")
            log_pipeline_comparison_audit(
                dataset_name=meta.get("file_name", "unknown"),
                comparison_results=comparison_results,
                fidelity_ref_reloaded=fid_ref,
            )
        except Exception as audit_exc:
            print(f"[WARN] Auditoría de comparación falló: {audit_exc}")

        c45_tree = None
        if (
            hasattr(self.metrics_comparator, "c45_tree")
            and self.metrics_comparator.c45_tree.tree_model
        ):
            c45_tree = self.metrics_comparator.c45_tree.tree_model

        progress("report", 90, "Generando informe...")
        try:
            detailed_report = self.metrics_comparator.generate_comparison_report()
            if detailed_report is None:
                detailed_report = "No fue posible generar informe detallado."
        except Exception as e:
            import traceback

            print(f"Error al generar informe: {e}\n{traceback.format_exc()}")
            detailed_report = f"Error al generar informe detallado: {str(e)}"

        progress("done", 100, "Completado")
        result_text = f"""Comparación automática completada

{detailed_report}

Use la pestaña Métricas para el detalle visual de fidelidad y precisión.
"""
        return {
            "success": True,
            "comparison_results": comparison_results,
            "detailed_report": detailed_report,
            "c45_tree": c45_tree,
            "result_text": result_text,
        }

    def _validate_data_for_comparison(self, X, y):

        issues = []
        import numpy as np
        
        # Verificar tamanho mínimo
        if len(X) < 30:
            issues.append(f"Pocos datos ({len(X)} muestras). "
                         "Los resultados pueden ser inestables.")
        
        # Verificar número mínimo de clases
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            issues.append("Se necesitan al menos 2 clases para la comparación.")
        
        # Verificar desbalanceamento de clases
        if len(unique_classes) >= 2:
            class_counts = [np.sum(y == cls) for cls in unique_classes]
            max_count = max(class_counts)
            min_count = min(class_counts)
            
            if min_count == 0:
                issues.append("Alguna clase no tiene ejemplos.")
            else:
                imbalance_ratio = max_count / min_count
                if imbalance_ratio > 10:
                    issues.append(f"Clases muy desbalanceadas (ratio: {imbalance_ratio:.1f}:1). "
                                 "Considere usar muestreo estratificado.")
                elif imbalance_ratio > 5:
                    issues.append(f"Clases moderadamente desbalanceadas (ratio: {imbalance_ratio:.1f}:1).")
        
        # Verificar dimensões
        if len(X.shape) != 2:
            issues.append(f"Los datos deben ser 2D, pero se recibieron {len(X.shape)}D.")
        
        # Verificar se há features vazias ou constantes
        if X.shape[1] > 0:
            try:
                # Garantir que X é numérico antes de calcular estatísticas
                # Tenta converter para float, se houver strings, captura a exceção
                X_numeric = np.array(X, dtype=float)
                feature_stds = np.std(X_numeric, axis=0)
                constant_features = np.sum(feature_stds < 1e-10)
                if constant_features > 0:
                    issues.append(f"{constant_features} feature(s) son constantes "
                                 "(varianza cero). Pueden eliminarse.")
            except (ValueError, TypeError):
                # Se não conseguir converter para numérico, ignora esta verificação
                # Isso pode acontecer se os dados ainda não foram codificados
                pass
        
        return issues
            
    def show_natural_explanations(self):
        
        if not self.mlp_model:
            QMessageBox.warning(self, "Advertencia", "¡Entrene un modelo primero!")
            return
            
        try:
            meta = self.trepan.mlp_trainer.arff_meta
            if meta is None:
                feature_names = [f"feature_{i}" for i in range(len(self.trepan.feature_encoders))]
                class_names = list(self.trepan.label_encoder.classes_)
                meta = {
                    'features': feature_names,
                    'classes': class_names,
                    'target': 'target',
                    'file_name': 'unknown',
                    'sample_count': 0
                }
            else:
                feature_names = meta['features']
                class_names = list(self.trepan.label_encoder.classes_)
            
            X, y = self._get_original_xy()
            
            explanation_widget = ExplanationWidget(
                tree_model=self.trepan.extractor.explainer_tree,
                feature_names=feature_names,
                class_names=class_names,
                training_data=X,
                training_labels=y
            )
            
            dialog = QDialog(self)
            dialog.setWindowTitle("🧠 Explicaciones en Lenguaje Natural")
            dialog.setModal(True)
            dialog.setMinimumSize(640, 480)
            dialog.resize(900, 700)
            
            layout = QVBoxLayout()
            layout.addWidget(explanation_widget)
            
            close_btn = QPushButton("Cerrar")
            close_btn.setStyleSheet("background-color: #e74c3c; color: white; font-weight: bold; padding: 10px;")
            close_btn.clicked.connect(dialog.accept)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec()
            
            self.show_results("✅ ¡Interfaz de explicaciones abierta!\n\n"
                             "🧠 Recursos disponibles:\n"
                             "• 📋 Resúmenes automáticos de las reglas\n"
                             "• 💡 Ejemplos concretos para cada regla\n"
                             "• ❓ Sistema de preguntas y respuestas\n"
                             "• 🔍 Análisis de muestras específicas\n\n"
                             "💡 ¡Use los botones en la interfaz para explorar las explicaciones!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al generar explicaciones naturales: {str(e)}")
            
    def export_results(self):
        if not self.mlp_model:
            QMessageBox.warning(self, "Advertencia", "¡Entrene un modelo primero!")
            return
            
        try:
            directory = QFileDialog.getExistingDirectory(self, "Seleccione directorio para guardar")
            
            if directory:
                from datetime import datetime
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                
                results_file = os.path.join(directory, f"resultados_biuri_{timestamp}.txt")
                with open(results_file, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.toPlainText())

                # Evidência científica auditável: relatório completo, hashes do
                # protocolo e diagnóstico por linha solicitado na auditoria V9.1.
                comparison_report_file = os.path.join(
                    directory, f"comparacao_cientifica_{timestamp}.txt"
                )
                detailed_report = self.metrics_comparator.generate_comparison_report()
                if detailed_report is None:
                    detailed_report = "No fue posible generar informe detallado."
                with open(comparison_report_file, 'w', encoding='utf-8') as f:
                    f.write(detailed_report)
                try:
                    import json
                    protocol_file = os.path.join(
                        directory, f"protocolo_bloqueado_{timestamp}.json"
                    )
                    with open(protocol_file, 'w', encoding='utf-8') as f:
                        json.dump(
                            (self.metrics_comparator.comparison_results or {}).get(
                                "protocol_audit", {}
                            ),
                            f, ensure_ascii=False, indent=2, default=str,
                        )
                    diagnostics_file = os.path.join(
                        directory, f"diagnostico_linhas_teste_{timestamp}.csv"
                    )
                    self.metrics_comparator.export_row_diagnostics(diagnostics_file)
                except Exception as audit_export_exc:
                    print(f"[WARN] Exportação da evidência de auditoria falhou: {audit_export_exc}")
                
                # Dados estruturados (métricas, relatório semântico, diagnóstico das árvores, manifesto,
                # configuração). A imagem da árvore é outra ação: "Exportar árvore".
                structured_note = ""
                try:
                    from gui.export_results import export_results as export_structured
                    from gui.result_builder import build_experiment_result
                    result = (self.audit.result if self.audit is not None and self.audit.result is not None
                              else build_experiment_result(self))
                    target = os.path.join(directory, f"experimento_{result.provenance.experiment_id}_{timestamp}")
                    export_structured(result, target)
                    structured_note = "\n" + tr("export.results_done", path=target)
                    if result.stale:
                        structured_note += "\n" + tr("export.stale_warning")
                except Exception as structured_exc:
                    import logging
                    logging.getLogger("biuri.gui").exception("Exportação estruturada falhou")
                    structured_note = f"\n{tr('error.export')} ({structured_exc})"

                if self.loaded_ontology and hasattr(self.trepan.extractor, 'save_domain_knowledge'):
                    try:
                        knowledge_file = os.path.join(directory, f"domain_knowledge_{timestamp}.pkl")
                        if self.trepan.extractor.save_domain_knowledge(knowledge_file):
                            export_msg = f"✅ ¡Resultados exportados con éxito!\n\n📁 Directorio: {directory}\n📄 Archivos guardados con marca de tiempo: {timestamp}\n💾 Conocimiento ontológico guardado en: domain_knowledge_{timestamp}.pkl{structured_note}"
                        else:
                            export_msg = f"✅ ¡Resultados exportados con éxito!\n\n📁 Directorio: {directory}\n📄 Archivos guardados con marca de tiempo: {timestamp}{structured_note}"
                    except Exception as e:
                        print(f"Error ao exportar conhecimento ontológico: {e}")
                        export_msg = f"✅ ¡Resultados exportados con éxito!\n\n📁 Directorio: {directory}\n📄 Archivos guardados con marca de tiempo: {timestamp}{structured_note}"
                else:
                    export_msg = f"✅ ¡Resultados exportados con éxito!\n\n📁 Directorio: {directory}\n📄 Archivos guardados con marca de tiempo: {timestamp}{structured_note}"
                
                self.show_results(export_msg)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error al exportar resultados: {str(e)}")
            
    def export_tree(self):
        """Exporta a árvore como imagem (PNG/SVG/PDF). Ação distinta de "Exportar resultados"."""
        tree = self.trepan_reloaded_tree if self.trepan_reloaded_tree is not None else self.trepan_original_tree
        if tree is None:
            QMessageBox.information(self, tr("export.tree_title"), tr("export.no_tree"))
            return
        path, selected = QFileDialog.getSaveFileName(self, tr("export.tree_title"), "arvore_biuri.png", tr("export.tree_filter"))
        if not path:
            return
        try:
            from gui.pyqt_tree_controls import export_tree_png
            fmt = os.path.splitext(path)[1].lstrip(".").lower() or "png"
            if fmt not in ("png", "svg", "pdf"):
                fmt = "png"
            base = os.path.splitext(path)[0]
            names = self._get_original_feature_names() if tree is self.trepan_original_tree else (
                self.trepan_reloaded_feature_names or self._get_original_feature_names())
            classes = list(self.trepan.label_encoder.classes_)
            saved = export_tree_png(tree, names, classes, base, fmt=fmt)
            self._set_status(tr("export.tree_done", path=saved))
        except Exception as exc:
            if self.audit is not None:
                self._show_structured_error(self.audit.on_failure(exc, what_key="error.export", where_key="export.tree_title",
                                                                  action_key="error.action.generic"))
            else:
                QMessageBox.critical(self, tr("error.title"), str(exc))

    def show_progress(self, message):
        # Waiting as status — do not wipe the results log for a spinner line
        self._set_status(message)

    def show_results(self, text, switch_tab=True):
        self.results_text.setPlainText(text)
        if switch_tab:
            self.content_tabs.setCurrentWidget(self.results_tab)

    def closeEvent(self, event):
        for worker in (
            getattr(self, "_training_worker", None),
            getattr(self, "_metrics_worker", None),
            getattr(self, "_cf_worker", None),
        ):
            if worker is not None and worker.isRunning():
                if hasattr(worker, "request_cancel"):
                    worker.request_cancel()
                worker.wait(5000)
        super().closeEvent(event)
    
    def _update_metrics_visualizer_data(self):
        try:
            if hasattr(self, 'metrics_widget') and self.metrics_widget is not None:
                data_for_viz = self._get_original_xy()
                if data_for_viz is not None:
                    if hasattr(self, 'arff_meta') and self.arff_meta is not None:
                        self.metrics_widget.update_data_info(data_for_viz, self.arff_meta)
        except Exception as e:
            print(f"Error ao atualizar visualizador de métricas: {e}")


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(theme.APP_QSS)

    window = BiuriApp()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
