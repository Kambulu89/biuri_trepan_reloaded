"""Esquema do dataset na GUI: resolução automática (ARFF › ontologia › inferência) e validação antes do treino.

Sem configuração manual. A lógica de decisão está em ``core.dataset_schema`` e ``core.ontology_schema`` (agnósticas);
aqui só a integração com o estado da aplicação (modelo treinado, ontologia ativa, ficheiro carregado).
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QMessageBox

from core.dataset_schema import (
    DatasetSchema, ValidationReport, apply_schema, build_schema, file_fingerprint, move_target_last, read_arff_header,
    read_tabular_text, save_schema, validate_schema,
)
from core.ontology_schema import OntologySchemaDeclaration, range_kind, read_ontology_schema

log = logging.getLogger("biuri.gui.dataset_schema")


class DatasetSchemaMixin:
    """Métodos de BiuriApp para o esquema do dataset (sem __init__; o estado é lido com getattr)."""

    # ----------------------------------------------------------------- estado treinado / ontologia
    def _trained_snapshot(self, dataset_id: Optional[str] = None) -> Optional[dict]:
        """Encoder/modelo já treinado. Só protege o MESMO dataset: um modelo de outro ficheiro é irrelevante (o próximo treino
        substitui-o) e não pode bloquear dados novos. A identidade do dataset treinado é registada quando o treino arranca
        (``trainer.arff_meta`` é substituído a cada carregamento e não serve para isso)."""
        trainer = getattr(getattr(self, "trepan", None), "mlp_trainer", None)
        if trainer is None or getattr(trainer, "model", None) is None:
            return None
        rec = getattr(self, "_training_record", None) or {}
        if dataset_id is not None and rec.get("dataset_id") is not None and rec["dataset_id"] != dataset_id:
            return None
        raw_classes = getattr(getattr(trainer, "label_encoder", None), "classes_", None)
        classes = [] if raw_classes is None else list(raw_classes)       # classes_ é ndarray: nunca usar `or`
        return {"class_order": [str(c) for c in (classes or rec.get("class_order") or [])], "target": rec.get("target"),
                "feature_names": list(rec.get("feature_names") or []), "dataset_id": rec.get("dataset_id")}

    def _ontology_declaration(self, columns) -> Optional[OntologySchemaDeclaration]:
        """Esquema declarado na ontologia ATIVA (None sem ontologia). Cache por (ontologia, colunas)."""
        onto = getattr(self, "loaded_ontology", None)
        if onto is None:
            return None
        key = (id(onto), tuple(sorted(columns)))
        cache = getattr(self, "_ontology_schema_cache", None)
        if cache is None or cache[0] != key:
            cache = (key, read_ontology_schema(onto, list(columns)))
            self._ontology_schema_cache = cache
        return cache[1]

    def _ontology_type_info(self, schema: DatasetSchema) -> Optional[dict]:
        onto = getattr(self, "loaded_ontology", None)
        if onto is None:
            return None
        try:
            from core.ontology_quality import ontology_entities
            by_name = {getattr(e, "name", ""): e for group in ontology_entities(onto).values() for e in group}
            info = {}
            for d in self.ontology_quality_gate.match_features(schema.feature_names, onto):
                data = dict(d.__dict__)
                if data.get("accepted", True) and data.get("entity_name"):
                    info[str(data["feature"])] = {"entity": data["entity_name"], "range": range_kind(by_name.get(data["entity_name"]))}
            self._ontology_check_error = None
            return info
        except Exception as exc:  # a verificação de compatibilidade nunca impede o carregamento
            log.warning("Compatibilidade com a ontologia não verificada: %s", exc)
            self._ontology_check_error = str(exc)
            return None

    # ----------------------------------------------------------------- resolução e validação
    def _prepare_dataset_schema(self, file_path: str, df: pd.DataFrame, source_format: str, *, arff_header=None, tabular_meta=None):
        """Resolve e valida o esquema. Devolve (df para o pipeline, esquema). ARFF por omissão: df intacto."""
        cols = [str(c) for c in df.columns]
        dups = list((tabular_meta or {}).get("duplicate_columns") or [])
        fid = file_fingerprint(file_path)

        def make() -> DatasetSchema:
            kw = dict(file_path=file_path, source_format=source_format, arff_header=arff_header,
                      ontology_decl=self._ontology_declaration(cols),
                      separator=(tabular_meta or {}).get("separator"), encoding=(tabular_meta or {}).get("encoding"))
            probe = build_schema(df, **kw)
            trained = self._trained_snapshot(fid)
            # a ordem do modelo treinado só protege o MESMO alvo
            if trained and trained.get("target") == probe.target:
                kw["trained_class_order"] = trained.get("class_order")
            return build_schema(df, **kw)

        def validate(sch: DatasetSchema) -> ValidationReport:
            return validate_schema(sch, df, trained=self._trained_snapshot(fid), ontology_decl=self._ontology_declaration(cols),
                                   ontology_info=self._ontology_type_info(sch), duplicate_columns=dups)

        schema = make()
        self._dataset_raw_df = df
        self._dataset_make_schema = make
        self._dataset_validate = validate
        self.dataset_schema = schema
        self.dataset_validation = validate(schema)
        # o esquema resolvido (não trivial) fica registado para reproduzir/auditar a experiência
        if not schema.is_default_arff() and os.environ.get("BIURI_DATASET_CONFIG_AUTOSAVE", "1") != "0":
            try:
                save_schema(schema)
            except OSError as exc:
                log.warning("Não foi possível registar o esquema do dataset: %s", exc)
        if schema.is_default_arff():
            return df, schema
        out = move_target_last(df, schema.target)
        if source_format != "arff" or any(c.source == "ontology" for c in schema.columns):
            out = apply_schema(df, schema)
        return out, schema

    def _load_tabular_dataset(self, file_path: str):
        enc = getattr(self, "_dataset_encoding", None) or "utf-8"
        raw, meta = read_tabular_text(file_path, encoding=enc)
        out, schema = self._prepare_dataset_schema(file_path, raw, "csv", tabular_meta=meta)
        attributes = [{"name": c.name, "type": c.kind} for c in schema.columns if c.name != schema.target] + [
            {"name": schema.target, "type": "{" + ",".join(schema.class_order) + "}"}]
        relation = os.path.splitext(os.path.basename(file_path))[0]
        return out, relation, attributes, schema

    def _schema_meta_fields(self, schema: Optional[DatasetSchema]) -> Dict[str, Any]:
        if schema is None:
            return {}
        decl = self._ontology_declaration(schema.feature_names + [schema.target])
        return {
            "dataset_schema": schema.to_dict(), "dataset_schema_signature": schema.signature(),
            "target_source": schema.target_source, "class_mapping": dict(schema.class_mapping),
            "column_types": {c.name: c.kind for c in schema.columns},
            "forced_categorical_columns": list(schema.forced_categorical),
            "ontology_schema": decl.to_dict() if decl is not None else None,
        }

    def dataset_schema_summary(self) -> str:
        schema = getattr(self, "dataset_schema", None)
        if schema is None:
            return ""
        lines = list(schema.summary_lines())
        decl = self._ontology_declaration(schema.feature_names + [schema.target])
        if decl is not None:
            lines += decl.summary_lines()
        text = "\n".join(lines)
        rep = getattr(self, "dataset_validation", None)
        if rep is not None and rep.issues:
            text += "\n\n🔎 Validação do dataset:\n" + rep.format()
        return text

    # ----------------------------------------------------------------- ontologia mudou → re-resolver
    def _refresh_dataset_schema_after_ontology_change(self):
        """Ontologia carregada/removida depois dos dados: recalcula o esquema; se mudar, recarrega o ficheiro."""
        raw = getattr(self, "_dataset_raw_df", None)
        sch = getattr(self, "dataset_schema", None)
        path = (getattr(self, "dataset_info", None) or {}).get("path")
        make = getattr(self, "_dataset_make_schema", None)
        if raw is None or sch is None or not path or make is None:
            return
        try:
            new = make()
        except Exception as exc:
            log.warning("Esquema não re-resolvido após mudança de ontologia: %s", exc)
            return
        if new.signature() != sch.signature():
            self.load_data_from_file(path)
        else:
            self.dataset_schema = new
            self.dataset_validation = self._dataset_validate(new)

    def _schedule_schema_refresh(self):
        if getattr(self, "_dataset_raw_df", None) is not None:
            QTimer.singleShot(0, self._refresh_dataset_schema_after_ontology_change)

    # ----------------------------------------------------------------- validação pré-treino
    def _validate_dataset_before_training(self) -> bool:
        """Revalida contra o estado atual (modelo treinado, ontologia ativa). Bloqueia com incompatibilidades críticas."""
        schema = getattr(self, "dataset_schema", None)
        raw = getattr(self, "_dataset_raw_df", None)
        if schema is None or raw is None:
            return True
        try:
            rep = self._dataset_validate(self._dataset_make_schema())
        except Exception as exc:  # nunca bloquear por falha do próprio verificador
            log.warning("Validação do dataset indisponível: %s", exc)
            return True
        self.dataset_validation = rep
        if rep.critical:
            QMessageBox.critical(
                self, "Dataset incompatível — treino bloqueado",
                "Foram detetadas incompatibilidades entre o ficheiro de dados, a ontologia e/ou o modelo treinado:\n\n"
                + rep.format() + "\n\n" + "\n".join(schema.summary_lines()[:2]))
            return False
        # o treino vai arrancar com este esquema: regista o que o modelo/encoder treinado vai representar
        self._training_record = {"dataset_id": schema.dataset_id, "target": schema.target,
                                 "feature_names": list(schema.feature_names), "class_order": list(schema.class_order)}
        return True
