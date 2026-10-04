"""Painel de auditoria científica (Qt). Camada fina: só mostra o que o apresentador devolve.

Não calcula nada: recebe um ``ExperimentResult`` e delega toda a formatação a
``gui.result_presenter``. Separadores: Resumo, Ontologia, Modelos, Árvores, Semântica,
Experiência e Registo (opcional, modo científico).
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QFrame, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
                             QPlainTextEdit, QScrollArea, QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from core.experiment_result import ExperimentResult
from gui import result_presenter as rp
from gui.strings import term, tr

MODEL_KEYS = ("mlp_original", "mlp_ontological", "c45", "trepan_original", "trepan_reloaded")
METRIC_KEY_BY_HEADER_ORDER = {"predictive": rp.PREDICTIVE_METRICS, "fidelity": ("fidelity",), "complexity": rp.COMPLEXITY_KEYS}


class FitTable(QTableWidget):
    """Tabela que ajusta a altura ao conteúdo (sem barras de scroll internas) e re-quebra linhas ao redimensionar."""

    def _fit(self) -> None:
        self.resizeRowsToContents()
        header = self.horizontalHeader().height() if self.horizontalHeader().isVisible() else 0
        height = header + sum(self.rowHeight(r) for r in range(self.rowCount())) + 2 * self.frameWidth() + 2
        self.setFixedHeight(max(height, 28))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._fit()

    def showEvent(self, event):
        super().showEvent(event)
        self._fit()


def _prepare(table: QTableWidget) -> None:
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setWordWrap(True)
    table.setTextElideMode(Qt.TextElideMode.ElideNone)


def _kv_table(rows: Sequence[Tuple[str, str]]) -> QTableWidget:
    table = FitTable(len(rows), 2)
    _prepare(table)
    table.horizontalHeader().setVisible(False)
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
    table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
    for i, (k, v) in enumerate(rows):
        table.setItem(i, 0, QTableWidgetItem(k))
        table.setItem(i, 1, QTableWidgetItem(v))
    table._fit()
    return table


def _grid_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> QTableWidget:
    table = FitTable(len(rows), len(headers))
    _prepare(table)
    table.setHorizontalHeaderLabels(list(headers))
    table.verticalHeader().setVisible(False)
    table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
    table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    table.setAlternatingRowColors(True)
    header = table.horizontalHeader()
    header.setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
    for r, row in enumerate(rows):
        for c, value in enumerate(row):
            table.setItem(r, c, QTableWidgetItem(str(value)))
    table._fit()
    return table


def table_rows(table: QTableWidget) -> List[List[str]]:
    return [[(table.item(r, c).text() if table.item(r, c) else "") for c in range(table.columnCount())]
            for r in range(table.rowCount())]


def _section(title: str, widget: QWidget) -> QGroupBox:
    box = QGroupBox(title)
    lay = QVBoxLayout(box)
    lay.addWidget(widget)
    return box


class AuditPanel(QWidget):
    mode_changed = pyqtSignal(str)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.result: Optional[ExperimentResult] = None
        self.mode = rp.BASIC
        self._selected_only = False
        self._selected_tree = "trepan_original"
        self.setObjectName("auditPanel")

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        header = QHBoxLayout()
        self.state_label = QLabel("")
        self.state_label.setObjectName("auditStateLabel")
        self.build_label = QLabel("")
        self.build_label.setObjectName("auditBuildLabel")
        self.mode_combo = QComboBox()
        self.mode_combo.addItem(tr("mode.basic"), rp.BASIC)
        self.mode_combo.addItem(tr("mode.scientific"), rp.SCIENTIFIC)
        self.mode_combo.setToolTip(tr("mode.hint"))
        self.mode_combo.currentIndexChanged.connect(self._on_mode)
        header.addWidget(self.state_label, 1)
        header.addWidget(self.build_label)
        header.addWidget(self.mode_combo)
        root.addLayout(header)

        self.stale_banner = self._banner("#b71c1c", "#ffebee")
        self.cache_banner = self._banner("#0d47a1", "#e3f2fd")
        self.error_banner = self._banner("#b71c1c", "#ffebee")
        root.addWidget(self.stale_banner)
        root.addWidget(self.cache_banner)
        root.addWidget(self.error_banner)

        self.tabs = QTabWidget()
        self.tab_pages = {}   # widget que está no separador (área de scroll, ou a própria página do registo)
        self._page_layouts = {}
        for key in ("summary", "ontology", "models", "trees", "semantics", "experiment", "log"):
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(4, 4, 4, 4)
            self._page_layouts[key] = lay
            if key == "log":
                page = inner
            else:
                page = QScrollArea()
                page.setWidgetResizable(True)
                page.setFrameShape(QFrame.Shape.NoFrame)
                page.setWidget(inner)
            self.tab_pages[key] = page
            self.tabs.addTab(page, tr(f"tab.{key}"))
        root.addWidget(self.tabs, 1)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("auditLogView")
        self._page_layouts["log"].addWidget(self.log_view)
        self._tables = {}
        self.refresh()

    # ------------------------------------------------------------ utilitários
    @staticmethod
    def _banner(fg: str, bg: str) -> QLabel:
        lab = QLabel("")
        lab.setWordWrap(True)
        lab.setStyleSheet(f"QLabel{{color:{fg};background:{bg};border:1px solid {fg};border-radius:6px;padding:6px;font-weight:600;}}")
        lab.setVisible(False)
        return lab

    def _set_banner(self, label: QLabel, text: str) -> None:
        label.setText(text)
        label.setVisible(bool(text))

    def _clear_page(self, key: str) -> QVBoxLayout:
        lay = self._page_layouts[key]
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None and w is not self.log_view:
                w.setParent(None)
                w.deleteLater()
        return lay

    def _on_mode(self) -> None:
        self.mode = self.mode_combo.currentData()
        self.refresh()
        self.mode_changed.emit(self.mode)

    # ------------------------------------------------------------ API pública
    def set_state(self, text: str) -> None:
        self.state_label.setText(text)

    def set_error(self, text: str) -> None:
        self._set_banner(self.error_banner, text)

    def set_result(self, result: Optional[ExperimentResult]) -> None:
        self.result = result
        self.refresh()

    def set_log_lines(self, lines: Sequence[str]) -> None:
        self.log_view.setPlainText("\n".join(lines))

    def set_selected_only(self, flag: bool) -> None:
        self._selected_only = bool(flag)
        self.refresh()

    def select_tree(self, key: str) -> None:
        self._selected_tree = key
        self.refresh()

    def table(self, name: str) -> Optional[QTableWidget]:
        return self._tables.get(name)

    # ------------------------------------------------------------ renderização
    def refresh(self) -> None:
        self._tables = {}
        r = self.result
        self.tabs.setTabVisible(self.tabs.indexOf(self.tab_pages["log"]), self.mode == rp.SCIENTIFIC)
        if r is None:
            self._set_banner(self.stale_banner, "")
            self._set_banner(self.cache_banner, "")
            self.build_label.setText("")
            for key in ("summary", "ontology", "models", "trees", "semantics", "experiment"):
                lay = self._clear_page(key)
                lay.addWidget(QLabel(tr("summary.no_result")))
            return
        self.build_label.setText(rp.build_label(r.provenance.build))
        self._set_banner(self.stale_banner, rp.stale_banner(r))
        self._set_banner(self.cache_banner, rp.cache_banner(r))
        self._render_summary(r)
        self._render_ontology(r)
        self._render_models(r)
        self._render_trees(r)
        self._render_semantics(r)
        self._render_experiment(r)

    def _kv(self, name: str, rows) -> QTableWidget:
        t = _kv_table(rows)
        self._tables[name] = t
        return t

    def _grid(self, name: str, table_dict) -> QTableWidget:
        t = _grid_table(table_dict["headers"], table_dict["rows"])
        self._tables[name] = t
        return t

    def _render_summary(self, r):
        lay = self._clear_page("summary")
        lay.addWidget(self._kv("summary", rp.summary_rows(r)))
        lay.addWidget(_section(tr("section.dataset"), self._kv("dataset", rp.dataset_rows(r))))
        lay.addStretch(1)

    def _render_ontology(self, r):
        lay = self._clear_page("ontology")
        lay.addWidget(_section(tr("section.ontology"), self._kv("ontology", rp.ontology_rows(r))))
        lay.addWidget(_section(tr("section.enrichment"), self._kv("enrichment", rp.enrichment_rows(r))))
        lay.addStretch(1)

    def _render_models(self, r):
        lay = self._clear_page("models")
        tables = rp.metrics_tables(r)
        for group, title in (("predictive", "section.metrics.predictive"), ("fidelity", "section.metrics.fidelity"),
                             ("complexity", "section.metrics.complexity")):
            grid = self._grid(f"metrics_{group}", tables[group])
            self._decorate_metric_tooltips(grid, group, r)
            lay.addWidget(_section(tr(title), grid))
        for key in MODEL_KEYS:
            card = r.models.get(key)
            box = _section(term(key), self._kv(f"card_{key}", rp.model_card_rows(card, r, self.mode)))
            box.setObjectName(f"card_{key}")
            lay.addWidget(box)
        lay.addStretch(1)

    def _decorate_metric_tooltips(self, grid: QTableWidget, group: str, r: ExperimentResult) -> None:
        keys = METRIC_KEY_BY_HEADER_ORDER[group]
        order = {"predictive": 1, "fidelity": 2, "complexity": 1}[group]
        for c in range(order if group != "fidelity" else 2, grid.columnCount()):
            idx = c - (1 if group != "fidelity" else 2)
            metric = keys[idx] if idx < len(keys) else None
            if metric and grid.horizontalHeaderItem(c) is not None:
                grid.horizontalHeaderItem(c).setToolTip(rp.metric_tooltip(metric))
        model_by_label = {term(k): k for k in MODEL_KEYS}
        for row in range(grid.rowCount()):
            label = grid.item(row, 0).text() if grid.item(row, 0) else ""
            mkey = model_by_label.get(label)
            if not mkey:
                continue
            for c in range(1, grid.columnCount()):
                item = grid.item(row, c)
                if item is None:
                    continue
                idx = c - (1 if group != "fidelity" else 2)
                metric = keys[idx] if 0 <= idx < len(keys) else (keys[0] if keys else None)
                item.setToolTip(rp.provenance_text(r, mkey, metric))

    def _render_trees(self, r):
        lay = self._clear_page("trees")
        combo = QComboBox()
        for k in rp.TREE_KEYS:
            combo.addItem(term(k), k)
        combo.setCurrentIndex(max(0, combo.findData(self._selected_tree)))
        combo.currentIndexChanged.connect(lambda _=0: self.select_tree(combo.currentData()))
        lay.addWidget(combo)
        diag = r.trees.get(self._selected_tree)
        card = r.models.get(self._selected_tree)
        lay.addWidget(QLabel(f"{tr('field.oracle')}: {rp._oracle_text(card)}"))
        lay.addWidget(_section(tr("section.diagnostics"), self._kv("tree_diagnostics", rp.tree_diagnostic_rows(diag))))
        lay.addStretch(1)

    def _render_semantics(self, r):
        lay = self._clear_page("semantics")
        check = QCheckBox(tr("misc.selected_only"))
        check.setChecked(self._selected_only)
        check.toggled.connect(self.set_selected_only)
        lay.addWidget(check)
        lay.addWidget(_section(tr("section.semantic_features"),
                               self._grid("semantic_features", rp.semantic_features_table(r, self._selected_only))))
        lay.addWidget(_section(tr("section.semantic_splits"), self._grid("semantic_splits", rp.semantic_splits_table(r))))
        if r.controls:
            lay.addWidget(_section(tr("section.controls"), self._grid("controls", rp.controls_table(r))))
        if r.ablation:
            lay.addWidget(_section(tr("section.ablation"), self._grid("ablation", rp.ablation_table(r))))
        if r.benchmark is not None:
            lay.addWidget(_section(tr("section.benchmark"), self._grid("benchmark", rp.benchmark_table(r))))
        lay.addStretch(1)

    def _render_experiment(self, r):
        lay = self._clear_page("experiment")
        sci = rp.scientific_rows(r)
        if sci:
            lay.addWidget(_section(tr("section.scientific"), self._kv("scientific", sci)))
        lay.addWidget(self._kv("experiment", rp.experiment_rows(r)))
        lay.addStretch(1)
