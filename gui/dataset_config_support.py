"""Configuração e validação de datasets (CSV/ARFF) na GUI: alvo, ordem das classes, tipos, validação antes do treino.

Lógica de decisão em ``core.dataset_schema`` (agnóstica); aqui só a integração com o estado da aplicação e o diálogo.
Os fluxos programáticos (``load_data_from_file`` sem ``interactive``) nunca abrem diálogos.
"""
from __future__ import annotations

import datetime as _dt
import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QTextEdit, QVBoxLayout,
)

from core.dataset_schema import (
    KINDS, DatasetSchema, ValidationReport, apply_schema, build_schema, file_fingerprint, load_saved_schema,
    move_target_last, read_arff_header, read_tabular_text, save_schema, validate_schema,
)

log = logging.getLogger("biuri.gui.dataset_config")
_TABULAR_EXT = (".csv", ".tsv", ".txt")


def _range_kind(entity) -> Optional[str]:
    """Tipo de valor declarado (rdfs:range) de uma propriedade OWL → numeric/categorical/boolean/datetime (ou None)."""
    try:
        for item in list(getattr(entity, "range", []) or []):
            base = getattr(item, "base_datatype", item)
            if base in (int, float):
                return "numeric"
            if base is str:
                return "categorical"
            if base is bool:
                return "boolean"
            if base in (_dt.datetime, _dt.date):
                return "datetime"
    except Exception:
        return None
    return None


class DatasetConfigDialog(QDialog):
    """Seletor de alvo, ordem das classes e tipos; revalida em tempo real e só aceita sem incompatibilidades críticas."""

    def __init__(self, parent, raw_df: pd.DataFrame, make_schema, validate, schema: DatasetSchema, trained_info: Optional[dict]):
        super().__init__(parent)
        self.setWindowTitle("⚙ Configuração do dataset")
        self.resize(900, 700)
        self._raw = raw_df
        self._make_schema = make_schema      # (request) -> DatasetSchema
        self._validate = validate            # (schema, discard_trained) -> ValidationReport
        self._trained = trained_info
        self._request: Dict[str, Any] = {}
        self._schema = schema
        self._building = False

        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.target_combo = QComboBox()
        self.target_combo.addItems([str(c) for c in raw_df.columns])
        self.target_combo.setAccessibleName("Coluna-alvo")
        form.addRow("🎯 Coluna-alvo:", self.target_combo)
        layout.addLayout(form)

        row = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(QLabel("🏷️ Ordem das classes (índice interno = posição):"))
        self.class_list = QListWidget()
        self.class_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        left.addWidget(self.class_list)
        btns = QHBoxLayout()
        self.btn_up, self.btn_down, self.btn_auto = QPushButton("▲"), QPushButton("▼"), QPushButton("Ordem automática")
        for b in (self.btn_up, self.btn_down, self.btn_auto):
            btns.addWidget(b)
        left.addLayout(btns)
        row.addLayout(left, 1)
        right = QVBoxLayout()
        right.addWidget(QLabel("🧱 Tipos das colunas (corrija se necessário):"))
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(["Coluna", "Tipo", "Origem", "Ausentes", "Únicos", "Nota"])
        self.table.horizontalHeader().setStretchLastSection(True)
        right.addWidget(self.table)
        row.addLayout(right, 2)
        layout.addLayout(row, 1)

        self.drop_check = QCheckBox("Remover linhas com valores ausentes (fica registado na configuração)")
        layout.addWidget(self.drop_check)
        self.discard_check = QCheckBox("Descartar os modelos treinados e retreinar com esta configuração")
        self.discard_check.setVisible(bool(trained_info))
        layout.addWidget(self.discard_check)
        self.issues = QTextEdit(); self.issues.setReadOnly(True); self.issues.setMaximumHeight(160)
        layout.addWidget(self.issues)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.accepted.connect(self.accept); self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.target_combo.currentTextChanged.connect(self._on_target)
        self.btn_up.clicked.connect(lambda: self._move(-1)); self.btn_down.clicked.connect(lambda: self._move(1))
        self.btn_auto.clicked.connect(self._auto_order)
        self.drop_check.toggled.connect(lambda v: self._update({"drop_missing_rows": bool(v)}))
        self.discard_check.toggled.connect(lambda _v: self._refresh_report())
        self._populate(schema)

    # ----- estado
    def request(self) -> Dict[str, Any]:
        return dict(self._request)

    def discard_models(self) -> bool:
        return bool(self._trained) and self.discard_check.isChecked()

    def _kinds_from_table(self) -> Dict[str, Dict[str, Any]]:
        out: Dict[str, Dict[str, Any]] = {}
        for r in range(self.table.rowCount()):
            name = self.table.item(r, 0).text()
            combo = self.table.cellWidget(r, 1)
            spec = self._schema.column(name)
            if combo.currentText() != spec.kind or spec.source in ("manual", "saved"):
                entry: Dict[str, Any] = {"kind": combo.currentText()}
                if combo.currentText() == "ordinal":
                    entry["ordinal_order"] = spec.ordinal_order or sorted(spec.categories or [])
                out[name] = entry
        return out

    def _update(self, patch: Dict[str, Any], keep_types: bool = True):
        if self._building:
            return
        req = dict(self._request)
        req.update(patch)
        if keep_types and "type_overrides" not in patch:
            req["type_overrides"] = {**req.get("type_overrides", {}), **self._kinds_from_table()}
        self._request = req
        try:
            self._schema = self._make_schema(req)
        except Exception as exc:  # pedido inválido: mostra e mantém o esquema anterior
            self.issues.setPlainText(f"⛔ {exc}")
            self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(False)
            return
        self._populate(self._schema)

    def _populate(self, schema: DatasetSchema):
        self._building = True
        try:
            self.target_combo.setCurrentText(schema.target)
            self.class_list.clear()
            for c in schema.class_order:
                self.class_list.addItem(QListWidgetItem(str(c)))
            self.drop_check.setChecked(schema.drop_missing_rows)
            self.table.setRowCount(0)
            for spec in schema.columns:
                r = self.table.rowCount(); self.table.insertRow(r)
                item = QTableWidgetItem(spec.name); item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.table.setItem(r, 0, item)
                combo = QComboBox(); combo.addItems(list(KINDS)); combo.setCurrentText(spec.kind)
                combo.setEnabled(spec.name != schema.target)
                combo.currentTextChanged.connect(lambda _t: self._update({}))
                self.table.setCellWidget(r, 1, combo)
                for col, text in ((2, spec.source), (3, str(spec.n_missing)), (4, str(spec.n_unique)), (5, spec.note or ("a confirmar" if spec.ambiguous else ""))):
                    it = QTableWidgetItem(text); it.setFlags(it.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(r, col, it)
        finally:
            self._building = False
        self._refresh_report()

    def _refresh_report(self):
        rep = self._validate(self._schema, self.discard_models())
        text = rep.format() or "✅ Sem incompatibilidades."
        self.issues.setPlainText(text)
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setEnabled(rep.ok)

    # ----- ações
    def _on_target(self, name: str):
        if not self._building and name and name != self._schema.target:
            self._update({"manual_target": name, "manual_class_order": None, "type_overrides": {}}, keep_types=False)

    def _move(self, delta: int):
        row = self.class_list.currentRow()
        new = row + delta
        if row < 0 or not (0 <= new < self.class_list.count()):
            return
        item = self.class_list.takeItem(row); self.class_list.insertItem(new, item); self.class_list.setCurrentRow(new)
        self._update({"manual_class_order": [self.class_list.item(i).text() for i in range(self.class_list.count())]})

    def _auto_order(self):
        self._update({"manual_class_order": None})


class DatasetConfigMixin:
    """Métodos de BiuriApp para esquema do dataset (sem __init__; o estado é lido com getattr)."""

    # ----------------------------------------------------------------- estado treinado / ontologia
    def _trained_snapshot(self) -> Optional[dict]:
        trainer = getattr(getattr(self, "trepan", None), "mlp_trainer", None)
        if trainer is None or getattr(trainer, "model", None) is None:
            return None
        raw_classes = getattr(getattr(trainer, "label_encoder", None), "classes_", None)
        classes = [] if raw_classes is None else list(raw_classes)       # classes_ é ndarray: nunca usar `or`
        meta = getattr(trainer, "arff_meta", None) or {}
        return {"class_order": [str(c) for c in classes], "target": meta.get("target"),
                "feature_names": list(meta.get("original_features") or meta.get("features") or [])}

    def _ontology_type_info(self, schema: DatasetSchema) -> Optional[dict]:
        onto = getattr(self, "loaded_ontology", None)
        if onto is None:
            return None
        try:
            from core.ontology import ontology_entities
            by_name = {getattr(e, "name", ""): e for group in ontology_entities(onto).values() for e in group}
            info = {}
            for d in self.ontology_quality_gate.match_features(schema.feature_names, onto):
                data = dict(d.__dict__)
                if data.get("accepted", True) and data.get("entity_name"):
                    info[str(data["feature"])] = {"entity": data["entity_name"], "range": _range_kind(by_name.get(data["entity_name"]))}
            return info
        except Exception as exc:  # a verificação de compatibilidade nunca impede o carregamento
            log.warning("Compatibilidade com a ontologia não verificada: %s", exc)
            return None

    # ----------------------------------------------------------------- carregamento
    def _prepare_dataset_schema(self, file_path: str, df: pd.DataFrame, source_format: str, *, arff_header=None,
                                tabular_meta=None, interactive: bool = False):
        """Constrói/valida o esquema. Devolve (df para o pipeline, esquema). ARFF por omissão: df intacto."""
        request = dict(getattr(self, "_pending_schema_request", None) or {})
        self._pending_schema_request = None
        saved = load_saved_schema(file_fingerprint(file_path))
        trained = self._trained_snapshot()
        kw = dict(file_path=file_path, source_format=source_format, arff_header=arff_header, saved=saved,
                  trained_class_order=None,
                  separator=(tabular_meta or {}).get("separator"), encoding=(tabular_meta or {}).get("encoding"))
        # a ordem do modelo treinado só protege o MESMO alvo
        probe = build_schema(df, **{**kw, **request})
        if trained and trained.get("target") == probe.target:
            kw["trained_class_order"] = trained.get("class_order")
        make = lambda req: build_schema(df, **{**kw, **{**request, **req}})
        schema = make({})
        dups = list((tabular_meta or {}).get("duplicate_columns") or [])

        def validate(sch: DatasetSchema, discard_trained: bool = False) -> ValidationReport:
            return validate_schema(sch, df, trained=None if discard_trained else trained, saved=saved,
                                   ontology_info=self._ontology_type_info(sch), duplicate_columns=dups)

        self._dataset_raw_df = df
        self._dataset_make_schema = make
        self._dataset_validate = validate
        self._dataset_duplicate_columns = dups
        needs_dialog = interactive and source_format != "arff" and saved is None
        if needs_dialog:
            dlg = DatasetConfigDialog(self, df, lambda req: make(req), validate, schema, trained)
            if dlg.exec() != QDialog.DialogCode.Accepted:
                raise ValueError("Carregamento cancelado: configuração do dataset não confirmada.")
            if dlg.discard_models():
                self._discard_trained_artifacts()
                trained = None
                kw["trained_class_order"] = None
            req = dlg.request()
            schema = make(req)
            request = {**request, **req}
        self._dataset_request = request
        self.dataset_schema = schema
        self.dataset_validation = validate(schema)
        # um esquema não trivial (CSV, alvo/ordem/tipos manuais) fica persistido para reproduzir a experiência
        if not schema.is_default_arff() and os.environ.get("BIURI_DATASET_CONFIG_AUTOSAVE", "1") != "0":
            try:
                save_schema(schema)
            except OSError as exc:
                log.warning("Não foi possível guardar a configuração do dataset: %s", exc)
        if schema.is_default_arff():
            return df, schema
        out = move_target_last(df, schema.target)
        if source_format != "arff" or any(c.source in ("manual", "saved") for c in schema.columns) or schema.drop_missing_rows:
            out = apply_schema(df, schema)
        return out, schema

    def _load_tabular_dataset(self, file_path: str, *, interactive: bool = False):
        enc = getattr(self, "_dataset_encoding", None) or "utf-8"
        raw, meta = read_tabular_text(file_path, encoding=enc)
        out, schema = self._prepare_dataset_schema(file_path, raw, "csv", tabular_meta=meta, interactive=interactive)
        attributes = [{"name": c.name, "type": c.kind} for c in schema.columns if c.name != schema.target] + [
            {"name": schema.target, "type": "{" + ",".join(schema.class_order) + "}"}]
        relation = os.path.splitext(os.path.basename(file_path))[0]
        return out, relation, attributes, schema

    def _schema_meta_fields(self, schema: Optional[DatasetSchema]) -> Dict[str, Any]:
        if schema is None:
            return {}
        return {
            "dataset_schema": schema.to_dict(), "dataset_schema_signature": schema.signature(),
            "target_source": schema.target_source, "class_mapping": dict(schema.class_mapping),
            "column_types": {c.name: c.kind for c in schema.columns},
            "forced_categorical_columns": list(schema.forced_categorical),
        }

    def dataset_schema_summary(self) -> str:
        schema = getattr(self, "dataset_schema", None)
        if schema is None:
            return ""
        text = "\n".join(schema.summary_lines())
        rep = getattr(self, "dataset_validation", None)
        if rep is not None and rep.issues:
            text += "\n\n🔎 Validação do dataset:\n" + rep.format()
        return text

    # ----------------------------------------------------------------- reconfiguração e validação pré-treino
    def open_dataset_config_dialog(self):
        schema = getattr(self, "dataset_schema", None)
        raw = getattr(self, "_dataset_raw_df", None)
        if schema is None or raw is None:
            QMessageBox.warning(self, "Advertencia", "¡Cargue datos primero!")
            return
        if self._warn_if_busy("configurar el dataset"):
            return
        dlg = DatasetConfigDialog(self, raw, self._dataset_make_schema, self._dataset_validate, schema, self._trained_snapshot())
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if dlg.discard_models() or self._has_any_trained_model():
            if not dlg.discard_models():
                ans = QMessageBox.question(
                    self, "Modelos treinados",
                    "Alterar a configuração do dataset descarta os modelos e árvores já treinados.\n\nContinuar?")
                if ans != QMessageBox.StandardButton.Yes:
                    return
            self._discard_trained_artifacts()
        self._pending_schema_request = {**getattr(self, "_dataset_request", {}), **dlg.request()}
        self.load_data_from_file(self.dataset_info["path"] if getattr(self, "dataset_info", None) else self.arff_meta["file_name"])

    def _validate_dataset_before_training(self) -> bool:
        """Revalida contra o estado atual (modelo, ontologia). Bloqueia com incompatibilidades críticas."""
        schema = getattr(self, "dataset_schema", None)
        raw = getattr(self, "_dataset_raw_df", None)
        if schema is None or raw is None:
            return True
        try:
            rep = self._dataset_validate(schema, False)
        except Exception as exc:  # nunca bloquear por falha do próprio verificador
            log.warning("Validação do dataset indisponível: %s", exc)
            return True
        self.dataset_validation = rep
        if rep.critical:
            QMessageBox.critical(
                self, "Dataset incompatível — treino bloqueado",
                "Corrija as incompatibilidades antes de treinar (⚙ Configurar dataset):\n\n" + rep.format()
                + "\n\n" + "\n".join(schema.summary_lines()[:2]))
            return False
        return True
