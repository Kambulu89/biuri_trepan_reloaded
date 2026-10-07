"""Painel PyQt6 para geração e transferência contrafactual."""
from __future__ import annotations

from html import escape
from numbers import Number
from typing import Any, Dict, Mapping, Sequence

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gui import theme


class CounterfactualPanel(QWidget):
    generate_requested = pyqtSignal(dict)
    global_requested = pyqtSignal(dict)
    tree_requested = pyqtSignal(dict)
    visualize_tree_requested = pyqtSignal(dict)
    transfer_requested = pyqtSignal(dict)
    export_requested = pyqtSignal(dict)

    MODEL_NAMES = (
        "MLP Original",
        "MLP Ontológica",
        "Trepan Original",
        "Trepan Reloaded",
        "C4.5-Nativo",
    )
    METHODS = (
        ("Automático", "AUTO"),
        ("DiCE", "DICE"),
        ("CLEAR", "CLEAR"),
        ("CoGS", "COGS"),
        ("LORE-Local", "LORE-LOCAL"),
        ("LORE-Global", "LORE-GLOBAL"),
    )
    METRIC_LABELS = {
        "validity": "Validade",
        "proximity": "Proximidade normalizada",
        "sparsity": "Features alteradas",
        "diversity": "Diversidade",
        "stability": "Estabilidade",
        "robustness": "Robustez local",
        "plausibility": "Plausibilidade",
        "actionability": "Acionabilidade",
        "ontology_consistency": "Consistência ontológica",
        "causal_consistency": "Consistência de regras de domínio (não causal)",
        "plausibility_score": "Plausibilidade (percentil kNN no treino)",
        "global_fidelity": "Fidelidade global",
        "local_fidelity": "Fidelidade local",
        "rule_coverage": "Cobertura das regras",
        "instance_coverage": "Cobertura das instâncias",
        "mean_conditions": "Média de condições por regra",
        "max_conditions": "Máximo de condições por regra",
        "mean_symbolic_changes": "Média de alterações simbólicas",
        "min_symbolic_changes": "Mínimo de alterações simbólicas",
        "max_symbolic_changes": "Máximo de alterações simbólicas",
        "symbolic_stability_proxy": "Estabilidade simbólica (proxy)",
        "rule_consistency": "Consistência das regras",
        "n_rules": "Regras extraídas",
        "n_counterfactual_rules": "Regras contrafactuais",
        "n_filtered_fragile": "Regras frágeis filtradas",
        "n_clusters": "Grupos de regras",
        "fidelity_to_oracle": "Fidelidade ao oráculo",
        "counterfactual_fidelity": "Fidelidade nos contrafactuais",
        "factual_fidelity": "Fidelidade na instância factual",
        "fidelity_to_original_tree": "Fidelidade à árvore original",
        "depth": "Profundidade",
        "n_nodes": "Nós",
        "n_leaves": "Folhas",
        "n_reference_neighbors": "Vizinhos de referência",
        "n_training_neighbors": "Vizinhos de treino",
        "n_evaluation_neighbors": "Vizinhos de avaliação",
        "n_valid_counterfactuals": "Contrafactuais válidos",
        "evaluation_mode": "Modo de avaliação",
        "n_candidates": "Candidatos avaliados",
        "mean_validity": "Validade média",
        "mean_proximity_l1": "Distância L1 média",
        "mean_proximity_l2": "Distância L2 média",
        "mean_proximity_weighted_l1": "Distância L1 ponderada média",
        "mean_sparsity_l0": "Features alteradas (L0)",
        "mean_sparsity_normalized": "Esparsidade normalizada",
        "mean_plausibility_nn_target": "Distância ao vizinho da classe alvo",
        "mean_plausibility_mahalanobis": "Distância de Mahalanobis",
        "mean_probability_margin": "Margem probabilística",
        "mean_distance_to_boundary": "Distância à fronteira",
        "diversity_weighted_l2": "Diversidade L2 ponderada",
        "valid_mlp_cf_rate": "Taxa de CFs válidos no MLP",
        "mean_joint_robustness": "Robustez conjunta média",
    }
    PERCENT_METRICS = {
        "validity", "stability", "robustness", "plausibility", "plausibility_score", "actionability",
        "ontology_consistency", "causal_consistency", "global_fidelity",
        "local_fidelity", "rule_coverage", "instance_coverage",
        "symbolic_stability_proxy", "rule_consistency", "fidelity_to_oracle",
        "counterfactual_fidelity", "factual_fidelity", "fidelity_to_original_tree",
        "mean_validity", "mean_sparsity_normalized", "valid_mlp_cf_rate",
        "mean_joint_robustness",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_result: Dict[str, Any] = {}
        self._local_result: Dict[str, Any] = {}
        self._generation_ready = False
        self._global_ready = False
        self._tree_ready = False
        self._visualization_ready = False
        self._transfer_ready = False
        self._build()

    def _build(self) -> None:
        self.setObjectName("counterfactualPanel")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self.scroll_area = QScrollArea()
        self.scroll_area.setObjectName("cfScrollArea")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        canvas = QWidget()
        canvas.setObjectName("cfCanvas")
        self.canvas = canvas
        root = QVBoxLayout(canvas)
        root.setContentsMargins(12, 10, 12, 12)
        root.setSpacing(10)
        self.scroll_area.setWidget(canvas)
        outer.addWidget(self.scroll_area)

        scope_banner = QFrame()
        scope_banner.setObjectName("scopeBanner")
        scope_layout = QHBoxLayout(scope_banner)
        scope_layout.setContentsMargins(14, 9, 14, 9)
        scope_layout.setSpacing(10)
        self.dataset_status_dot = QLabel("●")
        self.dataset_status_dot.setObjectName("datasetStatusDot")
        self.dataset_status_dot.setAccessibleName("Estado do dataset")
        self.dataset_scope_label = QLabel(
            "Dataset ativo: nenhum dataset treinado."
        )
        self.dataset_scope_label.setObjectName("datasetScopeLabel")
        self.dataset_scope_label.setAccessibleName("Dataset da análise contrafactual")
        self.dataset_scope_label.setWordWrap(True)
        self.scope_badge = QLabel("APENAS DATASET CARREGADO")
        self.scope_badge.setObjectName("scopeBadge")
        self.scope_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        scope_layout.addWidget(self.dataset_status_dot)
        scope_layout.addWidget(self.dataset_scope_label, 1)
        scope_layout.addWidget(self.scope_badge)
        root.addWidget(scope_banner)

        controls = QGroupBox("Parâmetros da explicação")
        controls.setObjectName("cfControlsBox")
        self.controls_box = controls
        self.controls_grid = QGridLayout(controls)
        self.controls_grid.setContentsMargins(12, 20, 12, 10)
        self.controls_grid.setHorizontalSpacing(12)
        self.controls_grid.setVerticalSpacing(6)

        self.model_combo = QComboBox()
        self.model_combo.addItems(self.MODEL_NAMES)
        self.instance_spin = QSpinBox()
        self.instance_spin.setRange(0, 0)
        self.desired_combo = QComboBox()
        self.desired_combo.addItem("Automática (outra classe)", None)
        self.method_combo = QComboBox()
        for label, value in self.METHODS:
            self.method_combo.addItem(label, value)
        self.total_spin = QSpinBox()
        self.total_spin.setRange(1, 20)
        self.total_spin.setValue(5)
        self.robustness_samples = QSpinBox()
        self.robustness_samples.setRange(10, 1000)
        self.robustness_samples.setValue(100)
        self.robustness_epsilon = QDoubleSpinBox()
        self.robustness_epsilon.setRange(0.001, 0.25)
        self.robustness_epsilon.setDecimals(3)
        self.robustness_epsilon.setSingleStep(0.005)
        self.robustness_epsilon.setValue(0.02)
        self.owl_checkbox = QCheckBox("Validar coerência OWL quando aplicável")
        self.owl_checkbox.setChecked(True)
        self.rst_checkbox = QCheckBox("Aplicar filtro de relevância RST")
        self.show_all_checkbox = QCheckBox("Mostrar todas as features (por defeito só as alteradas)")
        self.show_all_checkbox.setToolTip("Apenas muda a apresentação: o contrafactual não é alterado")

        self.model_combo.setToolTip("Modelo cuja decisão será explicada")
        self.instance_spin.setToolTip("Índice da instância no dataset ativo")
        self.desired_combo.setToolTip("Classe que o contrafactual deve alcançar")
        self.method_combo.setToolTip("Método de geração compatível com o modelo alvo")
        self.total_spin.setToolTip("Número máximo de candidatos a apresentar")
        self.robustness_samples.setToolTip(
            "Quantidade de perturbações usadas para estimar a robustez local"
        )
        self.robustness_epsilon.setToolTip(
            "Amplitude das perturbações aplicadas durante a avaliação de robustez"
        )

        validations = QFrame()
        validations.setProperty("fieldCell", True)
        validations_layout = QVBoxLayout(validations)
        validations_layout.setContentsMargins(2, 2, 2, 2)
        validations_layout.setSpacing(4)
        validation_label = QLabel("Validações opcionais")
        validation_label.setProperty("fieldLabel", True)
        validations_layout.addWidget(validation_label)
        validations_layout.addWidget(self.owl_checkbox)
        validations_layout.addWidget(self.rst_checkbox)
        validations_layout.addWidget(self.show_all_checkbox)

        self._configuration_cells = [
            self._field_cell("Modelo alvo", self.model_combo),
            self._field_cell("Instância", self.instance_spin),
            self._field_cell("Classe desejada", self.desired_combo),
            self._field_cell("Método", self.method_combo),
            self._field_cell("N.º de candidatos", self.total_spin),
            self._field_cell("Perturbações de robustez", self.robustness_samples),
            self._field_cell("Epsilon de robustez", self.robustness_epsilon),
            validations,
        ]

        for widget in (
            self.model_combo,
            self.instance_spin,
            self.desired_combo,
            self.method_combo,
            self.total_spin,
            self.robustness_samples,
            self.robustness_epsilon,
        ):
            widget.setMinimumHeight(36)
            widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        action_frame = QFrame()
        action_frame.setObjectName("cfActionBar")
        self.actions_grid = QGridLayout(action_frame)
        self.actions_grid.setContentsMargins(0, 0, 0, 0)
        self.actions_grid.setHorizontalSpacing(8)
        self.actions_grid.setVerticalSpacing(8)
        self.generate_button = QPushButton("✦  Gerar contrafactual")
        self.global_button = QPushButton("▦  Regras globais")
        self.tree_button = QPushButton("🌳  Construir árvore CF")
        self.transfer_button = QPushButton("⇄  Avaliar transferência")
        self.visualize_tree_button = QPushButton("◉  Visualizar árvore CF")
        self.export_button = QPushButton("⇩  Exportar")
        self.generate_button.setProperty("actionRole", "primary")
        self.global_button.setProperty("actionRole", "secondary")
        self.tree_button.setProperty("actionRole", "secondary")
        self.transfer_button.setProperty("actionRole", "secondary")
        self.visualize_tree_button.setProperty("actionRole", "secondary")
        self.export_button.setProperty("actionRole", "quiet")
        self.global_button.setToolTip(
            "Extrair todas as regras e transições de classe da árvore seleccionada"
        )
        self.tree_button.setToolTip(
            "Constrói a árvore explicativa CF (vizinhança + contrafactuais válidos). Se ainda não existirem contrafactuais locais para a "
            "instância, método e classe alvo escolhidos, gera-os primeiro; no fim abre a árvore na aba Visualização."
        )
        self.visualize_tree_button.setToolTip(
            "Mostrar a última árvore contrafactual na aba de visualização"
        )
        self.transfer_button.setToolTip(
            "Avaliar a transferência apenas no dataset actualmente carregado"
        )
        self.export_button.setToolTip("Exportar em JSON, CSV e Markdown")
        self.export_button.setEnabled(False)
        self.global_button.setEnabled(False)
        self.tree_button.setEnabled(False)
        self.visualize_tree_button.setEnabled(False)
        self._action_buttons = [
            self.generate_button, self.global_button, self.tree_button,
            self.transfer_button, self.visualize_tree_button, self.export_button,
        ]
        for button in self._action_buttons:
            button.setMinimumHeight(40)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        cards_box = QGroupBox("Indicadores principais")
        cards_box.setObjectName("metricCardsBox")
        self.cards_box = cards_box
        self.cards_grid = QGridLayout(cards_box)
        self.cards_grid.setContentsMargins(10, 20, 10, 10)
        self.cards_grid.setHorizontalSpacing(8)
        self.cards_grid.setVerticalSpacing(8)
        self.metric_cards = {}
        self.metric_card_titles = {}
        self._metric_card_widgets = []
        for index, (key, label) in enumerate((
            ("validity", "Validade"),
            ("proximity", "Proximidade"),
            ("sparsity", "Esparsidade"),
            ("robustness", "Robustez"),
            ("actionability", "Acionabilidade"),
            ("plausibility", "Plausibilidade"),
        )):
            title = QLabel(label)
            title.setProperty("metricTitle", True)
            value = QLabel("N/A")
            value.setAccessibleName(f"Métrica {label}")
            value.setProperty("metricValue", True)
            cell = QFrame()
            cell.setProperty("metricCard", True)
            cell.setMinimumHeight(72)
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(12, 8, 12, 8)
            cell_layout.setSpacing(2)
            cell_layout.addWidget(title)
            cell_layout.addWidget(value)
            self.metric_cards[key] = value
            self.metric_card_titles[key] = title
            self._metric_card_widgets.append(cell)

        top = QWidget()
        top_layout = QVBoxLayout(top)
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(10)
        top_layout.addWidget(controls)
        top_layout.addWidget(action_frame)
        top_layout.addWidget(cards_box)
        top.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        self.summary_text.setPlaceholderText(
            "Treine os modelos e gere uma explicação contrafactual."
        )
        self.summary_text.setAccessibleName("Resumo narrativo contrafactual")
        self.summary_text.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.summary_text.setMinimumWidth(260)
        self.summary_text.setMinimumHeight(160)

        self.metrics_table = QTableWidget(0, 2)
        self.metrics_table.setHorizontalHeaderLabels(["Métrica", "Valor"])
        self.metrics_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.metrics_table.setAccessibleName("Métricas contrafactuais")
        self.metrics_table.setMinimumHeight(120)
        self._configure_metrics_table()

        self.candidates_table = QTableWidget(0, 6)
        self.candidates_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.candidates_table.setAccessibleName("Candidatos contrafactuais")
        self.candidates_table.setMinimumHeight(150)
        self._set_candidate_mode("generation")
        self._configure_candidates_table()

        summary_box = QGroupBox("Explicação em linguagem natural")
        summary_box.setObjectName("summaryBox")
        self.summary_box = summary_box
        summary_layout = QVBoxLayout(summary_box)
        summary_layout.setContentsMargins(6, 12, 6, 6)
        summary_layout.addWidget(self.summary_text)

        metrics_box = QGroupBox("Métricas agregadas")
        metrics_box.setObjectName("metricsBox")
        self.metrics_box = metrics_box
        metrics_layout = QVBoxLayout(metrics_box)
        metrics_layout.setContentsMargins(6, 12, 6, 6)
        metrics_layout.addWidget(self.metrics_table)

        candidates_box = QGroupBox("Candidatos / resultados de transferência")
        candidates_box.setObjectName("candidatesBox")
        self.candidates_box = candidates_box
        candidates_layout = QVBoxLayout(candidates_box)
        candidates_layout.setContentsMargins(6, 12, 6, 6)
        candidates_layout.addWidget(self.candidates_table)

        result_tables = QSplitter(Qt.Orientation.Vertical)
        self.result_tables_splitter = result_tables
        result_tables.setChildrenCollapsible(False)
        result_tables.addWidget(metrics_box)
        result_tables.addWidget(candidates_box)
        result_tables.setStretchFactor(0, 2)
        result_tables.setStretchFactor(1, 3)
        result_tables.setHandleWidth(6)
        result_tables.setSizes([180, 230])

        lower = QSplitter(Qt.Orientation.Horizontal)
        self.results_splitter = lower
        lower.setChildrenCollapsible(False)
        lower.addWidget(summary_box)
        lower.addWidget(result_tables)
        lower.setStretchFactor(0, 2)
        lower.setStretchFactor(1, 4)
        lower.setHandleWidth(6)
        lower.setSizes([390, 820])
        lower.setMinimumHeight(300)

        root.addWidget(top)
        root.addWidget(lower, 1)

        self._apply_panel_style()
        self._reflow_layouts(1280)

        self.generate_button.clicked.connect(lambda: self.generate_requested.emit(self.options()))
        self.global_button.clicked.connect(lambda: self.global_requested.emit(self.options()))
        self.tree_button.clicked.connect(lambda: self.tree_requested.emit(self.options()))
        self.visualize_tree_button.clicked.connect(
            lambda: self.visualize_tree_requested.emit(dict(self.current_result))
        )
        self.transfer_button.clicked.connect(lambda: self.transfer_requested.emit(self.options()))
        self.export_button.clicked.connect(
            lambda: self.export_requested.emit(dict(self.current_result))
        )
        self.model_combo.currentTextChanged.connect(self._sync_methods)
        self._sync_methods(self.model_combo.currentText())

    def _field_cell(self, label_text: str, field: QWidget) -> QFrame:
        cell = QFrame()
        cell.setProperty("fieldCell", True)
        layout = QVBoxLayout(cell)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(4)
        label = QLabel(label_text)
        label.setProperty("fieldLabel", True)
        label.setBuddy(field)
        layout.addWidget(label)
        layout.addWidget(field)
        return cell

    def _apply_panel_style(self) -> None:
        self.setStyleSheet(f"""
            QWidget#cfCanvas {{
                background: {theme.SURFACE_RAISED};
                color: {theme.TEXT};
            }}
            QScrollArea#cfScrollArea {{
                background: {theme.SURFACE_RAISED};
                border: none;
            }}
            QFrame#scopeBanner {{
                background: #f6f0fa;
                border: 1px solid {theme.BORDER_SOFT};
                border-radius: {theme.RADIUS}px;
            }}
            QLabel#datasetStatusDot {{
                color: {theme.ACTION_LOAD};
                font-size: 16px;
            }}
            QLabel#datasetScopeLabel {{
                color: {theme.TEXT};
                font-size: 13px;
                font-weight: 600;
            }}
            QLabel#scopeBadge {{
                background: {theme.SURFACE_RAISED};
                color: {theme.BRAND_DARK};
                border: 1px solid {theme.BORDER_SOFT};
                border-radius: 10px;
                padding: 3px 9px;
                font-size: 10px;
                font-weight: 700;
            }}
            QGroupBox {{
                border: 1px solid {theme.BORDER};
                border-radius: {theme.RADIUS}px;
                margin-top: 10px;
                padding-top: 8px;
                font-weight: 600;
                background: {theme.SURFACE_RAISED};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                subcontrol-position: top left;
                left: 12px;
                padding: 0 6px;
                color: {theme.TEXT_SECONDARY};
                background: {theme.SURFACE_RAISED};
            }}
            QLabel[fieldLabel="true"], QLabel[metricTitle="true"] {{
                color: {theme.TEXT_SECONDARY};
                font-size: 11px;
                font-weight: 600;
            }}
            QComboBox, QSpinBox, QDoubleSpinBox {{
                min-height: 34px;
                background: {theme.SURFACE_RAISED};
                color: {theme.TEXT};
                border: 1px solid {theme.BORDER_CONTROL};
                border-radius: 6px;
                padding: 0 9px;
                selection-background-color: {theme.BRAND_LIGHT};
            }}
            QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
                border-color: {theme.BRAND_LIGHT};
            }}
            QComboBox::drop-down, QSpinBox::up-button, QSpinBox::down-button,
            QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
                width: 24px;
                background: {theme.SURFACE_CONTROL};
                border: none;
                border-left: 1px solid {theme.BORDER_CONTROL};
            }}
            QCheckBox {{
                color: {theme.TEXT};
                spacing: 7px;
                font-size: 11px;
            }}
            QFrame[metricCard="true"] {{
                background: {theme.SURFACE_SUBTLE};
                border: 1px solid {theme.BORDER_SOFT};
                border-radius: 7px;
            }}
            QLabel[metricValue="true"] {{
                color: {theme.BRAND_DARK};
                font-size: 18px;
                font-weight: 700;
            }}
            QPushButton {{
                min-height: 38px;
                border-radius: 6px;
                padding: 0 10px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton[actionRole="primary"] {{
                background: {theme.BRAND};
                color: {theme.TEXT_ON_BRAND};
                border: 1px solid {theme.BRAND};
            }}
            QPushButton[actionRole="primary"]:hover {{
                background: {theme.BRAND_DARK};
            }}
            QPushButton[actionRole="secondary"] {{
                background: {theme.SURFACE_RAISED};
                color: {theme.BRAND_DARK};
                border: 1px solid {theme.BRAND_LIGHT};
            }}
            QPushButton[actionRole="secondary"]:hover {{
                background: #f6eefa;
                border-color: {theme.BRAND};
            }}
            QPushButton[actionRole="quiet"] {{
                background: {theme.SURFACE_CONTROL};
                color: {theme.TEXT};
                border: 1px solid {theme.BORDER_CONTROL};
            }}
            QPushButton:disabled {{
                background: #f1f1f1;
                color: {theme.TEXT_DISABLED};
                border-color: {theme.BORDER};
            }}
            QTextEdit, QTableWidget {{
                background: {theme.SURFACE_RAISED};
                color: {theme.TEXT};
                border: 1px solid {theme.BORDER_CONTROL};
                border-radius: 5px;
                selection-background-color: #ead7f0;
                selection-color: {theme.TEXT};
            }}
            QTextEdit {{
                padding: 8px;
                font-size: 12px;
            }}
            QHeaderView::section {{
                background: {theme.SURFACE_SUBTLE};
                color: {theme.TEXT_SECONDARY};
                border: none;
                border-right: 1px solid {theme.BORDER};
                border-bottom: 1px solid {theme.BORDER};
                padding: 7px 6px;
                font-weight: 600;
            }}
            QTableWidget::item {{
                padding: 5px;
                border-bottom: 1px solid #eeeeee;
            }}
            QSplitter::handle {{
                background: {theme.BORDER};
                border-radius: 2px;
            }}
            QSplitter::handle:hover {{
                background: {theme.BRAND_LIGHT};
            }}
            QScrollBar:vertical {{
                width: 9px;
                background: transparent;
                margin: 2px;
            }}
            QScrollBar::handle:vertical {{
                min-height: 28px;
                background: #cfc3d4;
                border-radius: 4px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: {theme.BRAND_LIGHT};
            }}
        """)

    def _reflow_layouts(self, width: int) -> None:
        available = max(320, int(width) - 36)
        if available >= 1180:
            config_columns, action_columns, card_columns = 4, 6, 6
        elif available >= 780:
            config_columns, action_columns, card_columns = 2, 3, 3
        else:
            config_columns, action_columns, card_columns = 1, 2, 2
        self.scope_badge.setVisible(available >= 620)

        for cell in self._configuration_cells:
            self.controls_grid.removeWidget(cell)
        for index, cell in enumerate(self._configuration_cells):
            row, column = divmod(index, config_columns)
            self.controls_grid.addWidget(cell, row, column)
        for column in range(config_columns):
            self.controls_grid.setColumnStretch(column, 1)
        config_rows = (len(self._configuration_cells) + config_columns - 1) // config_columns
        self.controls_box.setMinimumHeight(24 + config_rows * 60)

        for button in self._action_buttons:
            self.actions_grid.removeWidget(button)
        for index, button in enumerate(self._action_buttons):
            row, column = divmod(index, action_columns)
            self.actions_grid.addWidget(button, row, column)
        for column in range(action_columns):
            self.actions_grid.setColumnStretch(column, 1)

        for card in self._metric_card_widgets:
            self.cards_grid.removeWidget(card)
        for index, card in enumerate(self._metric_card_widgets):
            row, column = divmod(index, card_columns)
            self.cards_grid.addWidget(card, row, column)
        for column in range(card_columns):
            self.cards_grid.setColumnStretch(column, 1)
        card_rows = (len(self._metric_card_widgets) + card_columns - 1) // card_columns
        self.cards_box.setMinimumHeight(28 + card_rows * 72)

        horizontal_results = available >= 900
        desired_orientation = (
            Qt.Orientation.Horizontal if horizontal_results else Qt.Orientation.Vertical
        )
        if self.results_splitter.orientation() != desired_orientation:
            self.results_splitter.setOrientation(desired_orientation)
            self.results_splitter.setSizes(
                [390, 820] if horizontal_results else [210, 390]
            )

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._reflow_layouts(event.size().width())

    def _configure_metrics_table(self) -> None:
        self.metrics_table.verticalHeader().setVisible(False)
        self.metrics_table.verticalHeader().setDefaultSectionSize(32)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setShowGrid(False)
        self.metrics_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.metrics_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.metrics_table.setWordWrap(False)
        self.metrics_table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.metrics_table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        header = self.metrics_table.horizontalHeader()
        header.setMinimumSectionSize(90)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def _configure_candidates_table(self) -> None:
        self.candidates_table.verticalHeader().setVisible(False)
        self.candidates_table.verticalHeader().setDefaultSectionSize(34)
        self.candidates_table.setAlternatingRowColors(True)
        self.candidates_table.setShowGrid(False)
        self.candidates_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.candidates_table.setSelectionMode(
            QAbstractItemView.SelectionMode.SingleSelection
        )
        self.candidates_table.setWordWrap(False)
        self.candidates_table.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.candidates_table.setVerticalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        header = self.candidates_table.horizontalHeader()
        header.setMinimumSectionSize(58)
        for column in (0, 1, 2, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

    def _set_candidate_mode(self, mode: str) -> None:
        if mode == "transfer":
            headers = ["Instância", "Método", "Categoria", "Mudança nas árvores",
                       "Proximidade", "Robustez conjunta"]
        elif mode == "global":
            headers = ["Regra factual", "Regra destino", "Classe", "Alterações",
                       "N.º mudanças", "Confiança"]
        elif mode == "tree":
            headers = ["Regra", "Classe", "Condições", "Descrição", "Suporte", "Confiança"]
        else:
            headers = ["#", "Método", "Classe", "Alterações", "Proximidade", "Robustez"]
        self.candidates_table.setHorizontalHeaderLabels(headers)

    def _sync_methods(self, model_name: str) -> None:
        current = self.method_combo.currentData()
        is_tree = model_name in {"Trepan Original", "Trepan Reloaded", "C4.5-Nativo"}
        allowed = (
            {"AUTO", "LORE-LOCAL", "LORE-GLOBAL", "COGS"}
            if is_tree else {"AUTO", "DICE", "CLEAR", "COGS"}
        )
        self.method_combo.blockSignals(True)
        self.method_combo.clear()
        for label, value in self.METHODS:
            if value in allowed:
                self.method_combo.addItem(label, value)
        index = self.method_combo.findData(current)
        self.method_combo.setCurrentIndex(index if index >= 0 else 0)
        self.method_combo.blockSignals(False)
        self._global_ready = self._generation_ready and is_tree
        if self._local_result:
            same_target = (
                str(self._local_result.get("target_model") or "") == str(model_name)
            )
            self._tree_ready = same_target and any(
                bool((item.get("metrics") or {}).get("validity"))
                for item in self._local_result.get("candidates") or []
            )
        self._update_action_states(False)

    def _update_action_states(self, busy: bool) -> None:
        self.generate_button.setEnabled(not busy and self._generation_ready)
        self.global_button.setEnabled(not busy and self._global_ready)
        self.tree_button.setEnabled(not busy and (self._tree_ready or self._generation_ready))
        self.transfer_button.setEnabled(not busy and self._transfer_ready)
        self.visualize_tree_button.setEnabled(not busy and self._visualization_ready)
        self.export_button.setEnabled(not busy and bool(self.current_result))

    def configure(
        self,
        *,
        n_instances: int,
        class_labels: Mapping[Any, Any],
        available_models: Sequence[str],
        dataset_name: str = "",
    ) -> None:
        has_dataset = bool(dataset_name and n_instances > 0)
        self.dataset_scope_label.setText(
            f"Dataset ativo: {dataset_name} · {int(n_instances)} instâncias disponíveis"
            if has_dataset else "Dataset ativo: nenhum dataset treinado."
        )
        self.dataset_scope_label.setToolTip(
            f"Todas as operações desta aba usam exclusivamente {dataset_name}."
            if has_dataset else "Carregue e treine um dataset para ativar esta aba."
        )
        self.dataset_status_dot.setStyleSheet(
            f"color: {theme.ACTION_LOAD if has_dataset else theme.TEXT_DISABLED};"
        )
        self.scope_badge.setText(
            "APENAS DATASET CARREGADO" if has_dataset else "AGUARDA TREINO"
        )
        self.instance_spin.setRange(0, max(0, int(n_instances) - 1))
        current = self.model_combo.currentText()
        self.model_combo.clear()
        for name in self.MODEL_NAMES:
            if name in set(available_models):
                self.model_combo.addItem(name)
        index = self.model_combo.findText(current)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.desired_combo.clear()
        self._multiclass = len(class_labels) > 2
        if self._multiclass:
            # Multiclasse: nunca se assume "classe oposta". O serviço devolve INVALID_TARGET com as opções válidas.
            self.desired_combo.addItem("Escolha a classe alvo (obrigatório)", None)
        else:
            self.desired_combo.addItem("Automática (classe oposta)", None)
        for value, label in class_labels.items():
            self.desired_combo.addItem(f"{label} ({value})", value)
        names = set(available_models)
        self._generation_ready = n_instances > 0 and self.model_combo.count() > 0
        self._transfer_ready = (
            n_instances > 0
            and {"Trepan Original", "Trepan Reloaded"}.issubset(names)
            and bool({"MLP Original", "MLP Ontológica"} & names)
        )
        self._global_ready = (
            self._generation_ready
            and self.model_combo.currentText() in {
                "Trepan Original", "Trepan Reloaded", "C4.5-Nativo"
            }
        )
        self._tree_ready = False
        self._visualization_ready = False
        self._local_result = {}
        self.current_result = {}
        self._last_generation = {}
        self.summary_text.clear()
        self.metrics_table.setRowCount(0)
        self.candidates_table.setRowCount(0)
        self._fill_cards({})
        self._update_action_states(False)

    def reset_results(self) -> None:
        """Limpa resultados mostrados (mantém a configuração) — usado quando dataset/modelo mudam."""
        self._tree_ready = False
        self._visualization_ready = False
        self._local_result = {}
        self.current_result = {}
        self._last_generation = {}
        self.summary_text.clear()
        self.metrics_table.setRowCount(0)
        self.candidates_table.setRowCount(0)
        self._fill_cards({})
        self._update_action_states(False)

    def options(self) -> Dict[str, Any]:
        return {
            "target_model": self.model_combo.currentText(),
            "instance_index": self.instance_spin.value(),
            "desired_class": self.desired_combo.currentData(),
            "method": self.method_combo.currentData(),
            "total_cfs": self.total_spin.value(),
            "validate_ontology": self.owl_checkbox.isChecked(),
            "apply_rst": self.rst_checkbox.isChecked(),
            "robustness_samples": self.robustness_samples.value(),
            "robustness_epsilon": self.robustness_epsilon.value(),
            "transfer_methods": ("LORE-LOCAL", "CLEAR", "COGS"),
            "fraction": 0.33,
            "global_max_per_rule": 5,
            "global_remove_fragile": True,
            "global_fragile_n_changes_threshold": 10,
            "cf_tree_max_depth": 5,
            "cf_tree_neighborhood_size": 300,
            "seed": 42,
            "max_time": 30.0,
            "max_iterations": 40,
            "pipeline": "cfkit",
            "show_all_features": self.show_all_checkbox.isChecked(),
        }

    def set_busy(self, busy: bool) -> None:
        self._update_action_states(busy)

    def display_result(self, result: Mapping[str, Any]) -> None:
        self.current_result = dict(result)
        result_type = result.get("result_type")
        runtime_tree = result.get("_runtime_tree_model")
        self._visualization_ready = (
            result_type == "counterfactual_tree"
            and runtime_tree is not None
            and (getattr(runtime_tree, "root_", None) is not None or getattr(runtime_tree, "tree_", None) is not None)
        )
        if result.get("rows") is not None:
            self._display_transfer(result)
        elif result_type == "global_rules":
            self._display_global(result)
        elif result_type == "counterfactual_tree":
            self._display_tree(result)
        else:
            self._local_result = dict(result)
            self._display_generation(result)
            self._tree_ready = any(
                bool((item.get("metrics") or {}).get("validity"))
                for item in result.get("candidates") or []
            )
        self._update_action_states(False)

    def _display_generation(self, result: Mapping[str, Any]) -> None:
        self._set_candidate_mode("generation")
        self._set_result_titles(
            "Explicação contrafactual",
            "Métricas agregadas e formais",
            "Candidatos contrafactuais",
        )
        self._set_card_labels({
            "validity": "Validade", "proximity": "Proximidade",
            "sparsity": "Esparsidade", "robustness": "Robustez",
            "actionability": "Acionabilidade", "plausibility": "Plausibilidade",
        })
        self._last_generation = dict(result)
        text = self._generation_text(result)
        self._set_summary_content("Resultado local", text)
        metrics_summary = result.get("aggregate_metrics") or {}
        formal_summary = (result.get("formal_evaluation") or {}).get("summary") or {}
        combined_metrics = dict(metrics_summary)
        combined_metrics.update({f"formal.{key}": value for key, value in formal_summary.items()})
        self._fill_metrics(combined_metrics)
        self._fill_cards(metrics_summary)
        candidates = result.get("candidates") or []
        self.candidates_table.setRowCount(len(candidates))
        for row_index, candidate in enumerate(candidates):
            metrics = candidate.get("metrics") or {}
            changed = "; ".join(_format_change(item) for item in candidate.get("changes") or [])
            values = (
                row_index + 1,
                candidate.get("method", ""),
                candidate.get("prediction", ""),
                changed,
                _format_number(metrics.get("proximity")),
                _format_percent(metrics.get("robustness")),
            )
            for column, value in enumerate(values):
                self._set_table_item(self.candidates_table, row_index, column, value)
        self._reset_result_scrolls()

    def _generation_text(self, result: Mapping[str, Any]) -> str:
        """Texto do resultado local: modelo explicado, estado, alterações (só as alteradas por defeito) e avisos."""
        lines = []
        if result.get("model_explained") or result.get("target_model"):
            lines.append(
                f"Modelo explicado: {result.get('model_explained') or result.get('target_model')}"
                f" · método: {result.get('method_label') or result.get('method')}"
                f" · estado: {result.get('status')}"
            )
        if result.get("message") and str(result.get("status")) not in {"SUCCESS", "success"}:
            lines.append(f"Motivo: {result['message']}")
        lines.append("")
        lines.append(result.get("narrative") or "Sem resumo disponível.")
        best = result.get("best_candidate")
        if best:
            lines.append("\nAlterações do melhor contrafactual:")
            lines += [f"  • {_format_change(item)}" for item in best.get("changes") or []] or ["  (nenhuma)"]
            if self.show_all_checkbox.isChecked() and result.get("original_human") and best.get("human"):
                lines.append("\nTodas as features (original → contrafactual):")
                changed = {item["feature"] for item in best.get("changes") or []}
                for name, original in result["original_human"].items():
                    cf_value = best["human"].get(name, original)
                    lines.append(f"  {'*' if name in changed else ' '} {name}: {original} → {cf_value}")
            if best.get("rule"):
                lines.append("\nRegra sugerida:\n" + str(best["rule"]))
            sem = (best.get("ontology_validation") or {})
            if sem.get("status"):
                lines.append(f"\nValidade semântica: {sem['status']} (a validade no modelo é independente)")
        warnings = result.get("warnings") or []
        if warnings:
            lines.append("\nAvisos:\n" + "\n".join(f"- {item}" for item in warnings))
        rejected = result.get("rejected") or []
        if rejected:
            lines.append(f"\nCandidatos rejeitados por constraints hard: {len(rejected)}")
        return "\n".join(lines)

    def _display_transfer(self, result: Mapping[str, Any]) -> None:
        self._set_candidate_mode("transfer")
        self._set_result_titles(
            "Interpretação da transferência",
            "Métricas de transferência",
            "Resultados de transferência",
        )
        self._set_card_labels({
            "validity": "CFs válidos", "proximity": "Proximidade",
            "sparsity": "Esparsidade", "robustness": "Robustez conjunta",
            "actionability": "Acionabilidade", "plausibility": "Plausibilidade",
        })
        dataset = result.get("dataset") or "dataset carregado"
        interpretation = str(result.get("interpretation") or "Transferência concluída.")
        self._set_summary_content(
            "Transferência no dataset ativo",
            f"Dataset analisado: {dataset}\nÂmbito: apenas o dataset carregado.\n\n"
            + interpretation
        )
        transfer_summary = result.get("summary") or {}
        self._fill_metrics(transfer_summary)
        self._fill_cards({
            "validity": transfer_summary.get("valid_mlp_cf_rate"),
            "proximity": None,
            "sparsity": None,
            "robustness": transfer_summary.get("mean_joint_robustness"),
            "actionability": None,
            "plausibility": None,
        })
        rows = result.get("rows") or []
        self.candidates_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.get("instance_index", ""),
                row.get("method", ""),
                row.get("category", ""),
                f"T={row.get('trepan_changed', '')}; R={row.get('reloaded_changed', '')}",
                _format_number(row.get("proximity")),
                _format_percent(row.get("joint_robustness")),
            )
            for column, value in enumerate(values):
                self._set_table_item(self.candidates_table, row_index, column, value)
        self._reset_result_scrolls()

    def _display_global(self, result: Mapping[str, Any]) -> None:
        self._set_candidate_mode("global")
        self._set_result_titles(
            "Síntese das regras globais",
            "Métricas globais",
            "Regras contrafactuais globais",
        )
        self._set_card_labels({
            "validity": "Cobertura de regras", "proximity": "Mudanças médias",
            "sparsity": "Condições médias", "robustness": "Estabilidade simbólica",
            "actionability": "Fidelidade global", "plausibility": "Consistência",
        })
        self._set_summary_content(
            "Análise global",
            str(result.get("narrative") or "Análise global concluída."),
        )
        metrics = result.get("aggregate_metrics") or {}
        self._fill_metrics(metrics)
        self._fill_cards({
            "validity": metrics.get("rule_coverage"),
            "proximity": metrics.get("mean_symbolic_changes"),
            "sparsity": metrics.get("mean_conditions"),
            "robustness": metrics.get("symbolic_stability_proxy"),
            "actionability": metrics.get("global_fidelity"),
            "plausibility": metrics.get("rule_consistency"),
        })
        rows = result.get("counterfactual_rules") or []
        self.candidates_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            changed = ", ".join(item.get("feature", "") for item in row.get("changes") or [])
            values = (
                row.get("factual_rule_id", ""), row.get("counterfactual_rule_id", ""),
                row.get("target_class_name", row.get("target_class", "")), changed,
                row.get("n_changes", ""), _format_percent(row.get("target_confidence")),
            )
            for column, value in enumerate(values):
                self._set_table_item(self.candidates_table, row_index, column, value)
        self._reset_result_scrolls()

    def _display_tree(self, result: Mapping[str, Any]) -> None:
        self._set_candidate_mode("tree")
        self._set_result_titles(
            "Síntese da árvore contrafactual",
            "Qualidade da árvore CF",
            "Regras da árvore explicativa",
        )
        self._set_card_labels({
            "validity": "Fidelidade oráculo", "proximity": "Profundidade",
            "sparsity": "N.º de folhas", "robustness": "Fidelidade CF",
            "actionability": "Fidelidade factual", "plausibility": "Fidelidade árvore",
        })
        self._set_summary_content(
            "Árvore explicativa contrafactual",
            str(result.get("narrative") or "Árvore CF concluída."),
        )
        metrics = result.get("aggregate_metrics") or {}
        self._fill_metrics(metrics)
        self._fill_cards({
            "validity": metrics.get("fidelity_to_oracle"),
            "proximity": metrics.get("depth"),
            "sparsity": metrics.get("n_leaves"),
            "robustness": metrics.get("counterfactual_fidelity"),
            "actionability": metrics.get("factual_fidelity"),
            "plausibility": metrics.get("fidelity_to_original_tree"),
        })
        rows = result.get("tree_rules") or []
        self.candidates_table.setRowCount(len(rows))
        for row_index, row in enumerate(rows):
            values = (
                row.get("rule_id", ""), row.get("predicted_class_name", ""),
                len(row.get("conditions") or []), row.get("rule", ""),
                _format_percent(row.get("support")), _format_percent(row.get("confidence")),
            )
            for column, value in enumerate(values):
                self._set_table_item(self.candidates_table, row_index, column, value)
        self._reset_result_scrolls()

    def _set_result_titles(self, summary: str, metrics: str, candidates: str) -> None:
        self.summary_box.setTitle(summary)
        self.metrics_box.setTitle(metrics)
        self.candidates_box.setTitle(candidates)

    def _set_summary_content(self, heading: str, text: str) -> None:
        safe_heading = escape(str(heading))
        safe_text = escape(str(text)).replace("\n", "<br>")
        self.summary_text.setHtml(
            f"<div style='font-family: Segoe UI; color: {theme.TEXT};'>"
            f"<div style='font-size: 15px; font-weight: 700; color: {theme.BRAND_DARK}; "
            f"margin-bottom: 8px;'>{safe_heading}</div>"
            f"<div style='font-size: 12px; line-height: 1.45;'>{safe_text}</div>"
            "</div>"
        )

    def _reset_result_scrolls(self) -> None:
        self.metrics_table.scrollToTop()
        self.candidates_table.scrollToTop()
        self.summary_text.verticalScrollBar().setValue(0)

    def _set_card_labels(self, labels: Mapping[str, str]) -> None:
        for key, text in labels.items():
            if key in self.metric_card_titles:
                self.metric_card_titles[key].setText(text)

    def _fill_metrics(self, metrics: Mapping[str, Any]) -> None:
        scalar_items = [
            (key, value) for key, value in metrics.items()
            if not isinstance(value, (dict, list, tuple))
        ]
        self.metrics_table.setRowCount(len(scalar_items))
        for row, (key, value) in enumerate(scalar_items):
            base_key = str(key).split(".")[-1]
            label = self.METRIC_LABELS.get(base_key, _humanize_metric_key(base_key))
            if str(key).startswith("formal."):
                label = f"Avaliação formal · {label}"
            self._set_table_item(self.metrics_table, row, 0, label)
            display = self._format_metric_value(base_key, value)
            self._set_table_item(self.metrics_table, row, 1, display)
        self.metrics_table.resizeRowsToContents()

    @staticmethod
    def _set_table_item(table: QTableWidget, row: int, column: int, value: Any) -> None:
        text = str(value)
        item = QTableWidgetItem(text)
        item.setToolTip(text)
        table.setItem(row, column, item)

    def _fill_cards(self, metrics: Mapping[str, Any]) -> None:
        percent_keys = {"validity", "robustness", "actionability", "plausibility"}
        for key, label in self.metric_cards.items():
            value = metrics.get(key)
            label.setText(
                _format_percent(value) if key in percent_keys else _format_number(value)
            )

    def _format_metric_value(self, key: str, value: Any) -> str:
        if value is None:
            return "Não aplicável"
        if isinstance(value, bool):
            return "Sim" if value else "Não"
        if key == "evaluation_mode":
            modes = {
                "holdout": "Holdout estratificado",
                "resubstitution_insufficient_local_classes": (
                    "Ressubstituição — classes locais insuficientes"
                ),
            }
            return modes.get(str(value), str(value))
        if isinstance(value, Number):
            numeric = float(value)
            if key in self.PERCENT_METRICS:
                return _format_percent(numeric)
            count_metric = (
                key.startswith("n_")
                or key in {"depth", "max_conditions", "min_symbolic_changes",
                           "max_symbolic_changes", "mean_sparsity_l0"}
            )
            if count_metric and numeric.is_integer():
                return f"{int(numeric):,}".replace(",", " ")
            return _format_number(numeric)
        return str(value)


def _format_change(item: Mapping[str, Any]) -> str:
    """'A: 10 → 13 (+3)'; categorias: 'B: X → Y'. Usa o espaço humano quando disponível."""
    before, after = item.get("original"), item.get("counterfactual")
    def fmt(v):
        return f"{v:.6g}" if isinstance(v, float) else str(v)
    delta = item.get("delta")
    suffix = f" ({delta:+.6g})" if isinstance(delta, (int, float)) and not isinstance(delta, bool) and item.get("kind") in (None, "continuous", "integer") else ""
    return f"{item.get('feature')}: {fmt(before)} → {fmt(after)}{suffix}"


def _format_number(value: Any) -> str:
    if value is None:
        return "—"
    numeric = float(value)
    if numeric.is_integer():
        return str(int(numeric))
    if abs(numeric) >= 1000:
        return f"{numeric:,.1f}".replace(",", " ")
    return f"{numeric:.4f}"


def _format_percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.1%}"


def _humanize_metric_key(key: str) -> str:
    return str(key).replace("_", " ").strip().capitalize()


__all__ = ["CounterfactualPanel"]
