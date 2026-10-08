"""Transformador ontológico stateful, ajustado exclusivamente no treino."""
from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from core.ontology_quality import OntologyQualityGate, annotation_values, entity_kind, ontology_entities
from core.ontology_semantic_graph import _explicit_entity_bounds


class OntologyNotFittedError(RuntimeError):
    pass


class OntologySchemaError(ValueError):
    pass


class OntologyProcessor:
    """``fit/transform`` para features OWL sem consultar validação/teste.

    O estado ajustado contém matching, limiares, grupos categóricos e o schema
    de saída. ``transform`` nunca inventa colunas nem usa ``fill_value=0``.
    """

    ONTO_PREFIX = "onto_"

    def __init__(
        self,
        ontology,
        matcher: Optional[Any] = None,
        *,
        quality_gate: Optional[OntologyQualityGate] = None,
        allow_train_calibrated_bounds: bool = False,
        drop_linear_redundant: bool = False,
        linear_redundancy_threshold: float = 0.999,
        near_duplicate_correlation: float = 0.9999,
        low_variance_rate: float = 0.999,
        reasoner_report: Optional[Dict[str, Any]] = None,
    ):
        self.ontology = ontology
        self.matcher = matcher
        self.quality_gate = quality_gate or OntologyQualityGate()
        self.allow_train_calibrated_bounds = bool(allow_train_calibrated_bounds)
        # Por omissão a redundância linear só é auditada; descartar é opt-in porque
        # os agregados de grupo são, por construção, combinações lineares.
        self.drop_linear_redundant = bool(drop_linear_redundant)
        self.linear_redundancy_threshold = float(linear_redundancy_threshold)
        # Filtros de novidade: acima destes limiares a feature não traz informação.
        self.near_duplicate_correlation = float(near_duplicate_correlation)
        self.low_variance_rate = float(low_variance_rate)
        self.reasoner_report = dict(reasoner_report or {})
        self._inferred_subclass_pairs = {
            (str(pair[0]), str(pair[1]))
            for pair in self.reasoner_report.get("inferred_subclass_relations", []) or []
            if len(pair) == 2
        }
        self.generation_summary_: Dict[str, Any] = {}
        self.is_fitted_ = False
        self.input_features_: List[str] = []
        self.output_features_: List[str] = []
        self.feature_specs_: List[Dict[str, Any]] = []
        self.column_entity_map_: Dict[str, str] = {}
        self.category_matching_: Dict[str, Dict[str, Any]] = {}
        self.feature_audit_: List[Dict[str, Any]] = []
        self.fit_row_count_: int = 0
        self.last_engineering_stats: Dict[str, Any] = {}

    def fit(
        self,
        dataframe: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_semantics: Optional[Dict[int, Dict]] = None,
        *,
        accepted_matches: Optional[Sequence[Dict[str, Any]]] = None,
        log: bool = True,
    ) -> "OntologyProcessor":
        if self.ontology is None:
            raise ValueError("Não é possível ajustar enriquecimento sem ontologia.")
        if dataframe is None or dataframe.empty:
            raise ValueError("O treino ontológico não pode ser vazio.")
        feature_cols = [str(c) for c in dataframe.columns if c != target_column]
        if len(set(feature_cols)) != len(feature_cols):
            raise OntologySchemaError("Features ARFF duplicadas antes do enriquecimento.")
        if any(name.startswith(self.ONTO_PREFIX) for name in feature_cols):
            raise OntologySchemaError(
                "fit exige apenas features ARFF originais; recebeu colunas onto_* pré-calculadas."
            )
        train = dataframe.loc[:, feature_cols].copy()
        self.input_features_ = feature_cols
        self.fit_row_count_ = len(train)
        if accepted_matches is None:
            decisions = self.quality_gate.match_features(feature_cols, self.ontology)
            accepted_matches = [decision.__dict__ for decision in decisions if decision.accepted]
        self.column_entity_map_ = {
            str(item["feature"]): str(item["entity_name"])
            for item in accepted_matches
            if item.get("accepted", True) and item.get("feature") in feature_cols
            and item.get("entity_name")
        }
        if feature_semantics:
            for info in feature_semantics.values():
                name = info.get("name")
                entity = info.get("matched_entity") or info.get("ontology_concept")
                if name in feature_cols and entity and not info.get("ontology_derived"):
                    self.column_entity_map_[name] = entity
        candidates: List[Dict[str, Any]] = []
        candidates.extend(self._fit_hierarchical_specs(train))
        candidates.extend(self._fit_relational_specs(train))
        candidates.extend(self._fit_constraint_specs(train))
        candidates.extend(self._fit_categorical_specs(train))
        self.feature_specs_ = self._deduplicate_specs(train, candidates)
        self.output_features_ = feature_cols + [spec["name"] for spec in self.feature_specs_]
        if len(set(self.output_features_)) != len(self.output_features_):
            raise OntologySchemaError("Enriquecimento produziu nomes de features duplicados.")
        self.is_fitted_ = True
        transformed = self.transform(train)
        inferred = [spec["name"] for spec in self.feature_specs_]
        by_kind = defaultdict(list)
        for spec in self.feature_specs_:
            by_kind[spec["kind"]].append(spec["name"])
        self.last_engineering_stats = {
            "fit_scope": "training_only",
            "fit_rows": self.fit_row_count_,
            "original_features": len(feature_cols),
            "mapped_direct": len(self.column_entity_map_),
            "hierarchical_features": len(by_kind["hierarchical_aggregate"]),
            "relational_features": len(by_kind["relational"]),
            "constraint_features": len(by_kind["constraint"]),
            "categorical_group_features": len(by_kind["categorical_group"]),
            "inferred_columns": inferred,
            "hierarchical_names": by_kind["hierarchical_aggregate"],
            "relational_names": by_kind["relational"],
            "constraint_names": by_kind["constraint"],
            "categorical_names": by_kind["categorical_group"],
            "category_matching": self.category_matching_,
            "feature_audit": list(self.feature_audit_),
            "total_training_features": transformed.shape[1],
            "dropped_duplicate_or_constant": len(candidates) - len(self.feature_specs_),
            "generation_summary": dict(self.generation_summary_),
            "statistical_feature_names": [
                row["feature"] for row in self.feature_audit_
                if row["accepted"] and row["knowledge_source"] == "statistical"
            ],
            "ontology_knowledge_feature_names": [
                row["feature"] for row in self.feature_audit_
                if row["accepted"] and row["knowledge_source"] == "ontology"
            ],
            "reasoner_inferred_feature_names": [
                row["feature"] for row in self.feature_audit_
                if row["accepted"] and row.get("reasoner_inferred")
            ],
            "linear_redundancy_threshold": self.linear_redundancy_threshold,
            "linear_redundant_features": [
                row["feature"] for row in self.feature_audit_
                if row.get("accepted")
                and row.get("linear_redundancy_r2") is not None
                and row["linear_redundancy_r2"] >= self.linear_redundancy_threshold
            ],
        }
        if log:
            self.log_feature_engineering_summary(self.last_engineering_stats)
        return self

    def merge_semantic_validation_audit(self, semantic_report: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combina proveniência OWL (fit) com efeito/estabilidade OOF (gate).

        O ``OntologyProcessor`` continua sem consultar rótulos durante ``fit``. Esta
        operação é chamada *depois* pelo pipeline de desenvolvimento e apenas
        acrescenta evidência de validação às linhas de auditoria já congeladas.
        """
        report = dict(semantic_report or {})
        semantic_rows = {
            str(row.get("feature")): row
            for row in (report.get("feature_audit") or [])
            if isinstance(row, dict) and row.get("feature")
        }
        merged: List[Dict[str, Any]] = []
        seen = set()
        for original in self.feature_audit_:
            row = dict(original)
            name = str(row.get("feature"))
            evidence = semantic_rows.get(name)
            if evidence:
                row["effect_on_metrics"] = evidence.get("effect_on_metrics")
                row["fold_stability"] = evidence.get("fold_stability")
                row["mean_mutual_information"] = evidence.get("mean_mutual_information")
                row["semantic_gate_accepted"] = bool(evidence.get("accepted_by_semantic_gate"))
                row["semantic_gate_rejection_reason"] = evidence.get("rejection_reason")
            else:
                row.setdefault("semantic_gate_accepted", False)
                row.setdefault("semantic_gate_rejection_reason", "not_evaluated_or_not_emitted")
                row.setdefault("mean_mutual_information", None)
            merged.append(row)
            seen.add(name)

        # Matriz enriquecida fornecida por um chamador externo pode conter
        # ``onto_*`` que não foram produzidas por este processor. Não inventar
        # proveniência: registar explicitamente como externa/desconhecida.
        for name, evidence in semantic_rows.items():
            if name in seen:
                continue
            merged.append({
                "feature": name,
                "kind": "external_semantic_feature",
                "owl_origin": {
                    "source_columns": [],
                    "owl_entities_or_properties": [],
                    "provenance": "external_or_unknown",
                },
                "derivation_rule": None,
                "provenance": "external_or_unknown",
                "accepted": None,
                "reason_not_duplicated": None,
                "rejection_reason": None,
                "duplicate_of": evidence.get("duplicate_of"),
                "constant_rate": evidence.get("constant_rate"),
                "effect_on_metrics": evidence.get("effect_on_metrics"),
                "fold_stability": evidence.get("fold_stability"),
                "mean_mutual_information": evidence.get("mean_mutual_information"),
                "semantic_gate_accepted": bool(evidence.get("accepted_by_semantic_gate")),
                "semantic_gate_rejection_reason": evidence.get("rejection_reason"),
            })
        self.feature_audit_ = merged
        if self.last_engineering_stats is not None:
            self.last_engineering_stats["feature_audit"] = list(merged)
            self.last_engineering_stats["semantic_gate_status"] = report.get("status")
            self.last_engineering_stats["semantic_gate_scope"] = report.get("scope")
        return list(merged)

    def transform(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted_:
            raise OntologyNotFittedError(
                "OntologyProcessor.transform chamado antes de fit no fold de treino."
            )
        if dataframe is None:
            raise OntologySchemaError("Matriz ausente no transform ontológico.")
        received = [str(c) for c in dataframe.columns]
        missing = [name for name in self.input_features_ if name not in received]
        unexpected = [name for name in received if name not in self.input_features_]
        if missing or unexpected:
            raise OntologySchemaError(
                f"Schema ontológico incompatível. Ausentes={missing}; inesperadas={unexpected}."
            )
        base = dataframe.loc[:, self.input_features_].copy()
        generated = {
            spec["name"]: self._apply_spec(base, spec) for spec in self.feature_specs_
        }
        result = (
            pd.concat([base, pd.DataFrame(generated, index=base.index)], axis=1)
            if generated else base
        )
        if list(result.columns) != self.output_features_:
            raise OntologySchemaError(
                "O transform não reproduziu exatamente o schema ajustado no treino."
            )
        return result

    def fit_transform(self, dataframe: pd.DataFrame, **kwargs) -> pd.DataFrame:
        target = kwargs.get("target_column")
        self.fit(dataframe, **kwargs)
        return self.transform(dataframe.drop(columns=[target], errors="ignore"))

    def apply_semantic_feature_engineering(
        self,
        dataframe: pd.DataFrame,
        target_column: Optional[str] = None,
        feature_semantics: Optional[Dict[int, Dict]] = None,
        *,
        fit: bool = False,
        accepted_matches: Optional[Sequence[Dict[str, Any]]] = None,
        log: bool = True,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Compatibilidade explícita; ``fit=True`` só deve ser usado no treino."""
        target = dataframe[[target_column]].copy() if target_column in dataframe.columns else None
        features = dataframe.drop(columns=[target_column], errors="ignore")
        if fit:
            self.fit(
                features,
                feature_semantics=feature_semantics,
                accepted_matches=accepted_matches,
                log=log,
            )
        transformed = self.transform(features)
        if target is not None:
            transformed = pd.concat([transformed, target], axis=1)
        return transformed, dict(self.last_engineering_stats)

    def enrich_training_matrix(
        self,
        X: np.ndarray,
        feature_names: List[str],
        feature_semantics: Optional[Dict[int, Dict]] = None,
        *,
        log: bool = True,
    ) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        frame = pd.DataFrame(np.asarray(X), columns=list(feature_names))
        self.fit(frame, feature_semantics=feature_semantics, log=log)
        enriched = self.transform(frame)
        return enriched.to_numpy(dtype=float), list(enriched.columns), dict(self.last_engineering_stats)

    def transform_matrix(self, X: np.ndarray, feature_names: Sequence[str]):
        frame = pd.DataFrame(np.asarray(X), columns=list(feature_names))
        enriched = self.transform(frame)
        return enriched.to_numpy(dtype=float), list(enriched.columns)

    def transform_fold_pair(
        self,
        train_frame: pd.DataFrame,
        validation_frame: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """Transforma um fold recalibrando apenas estatísticas train-only.

        O matching, a proveniência OWL e o schema de features continuam congelados
        pelo ``fit`` exterior. Parâmetros numéricos aprendidos (por exemplo média e
        desvio das agregações hierárquicas) são recalculados exclusivamente no
        sub-fold de treino. O estado do processor não é alterado.
        """
        if not self.is_fitted_:
            raise OntologyNotFittedError(
                "transform_fold_pair exige OntologyProcessor previamente ajustado."
            )
        train_frame = self._validate_base_frame(train_frame)
        validation_frame = self._validate_base_frame(validation_frame)
        fold_specs = self._recalibrate_specs(train_frame, self.feature_specs_)
        train_enriched = self._transform_with_specs(train_frame, fold_specs)
        validation_enriched = self._transform_with_specs(validation_frame, fold_specs)
        return train_enriched, validation_enriched, {
            "scope": "inner_fold_training_only",
            "fit_rows": int(len(train_frame)),
            "validation_rows": int(len(validation_frame)),
            "recalibrated_kinds": ["hierarchical_aggregate"],
            "schema": list(train_enriched.columns),
            "state_mutated": False,
        }

    def _validate_base_frame(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        if dataframe is None:
            raise OntologySchemaError("Matriz base ausente no fold ontológico.")
        received = [str(c) for c in dataframe.columns]
        missing = [name for name in self.input_features_ if name not in received]
        unexpected = [name for name in received if name not in self.input_features_]
        if missing or unexpected:
            raise OntologySchemaError(
                f"Schema base incompatível no fold. Ausentes={missing}; inesperadas={unexpected}."
            )
        return dataframe.loc[:, self.input_features_].copy()

    def _transform_with_specs(
        self, dataframe: pd.DataFrame, specs: Sequence[Dict[str, Any]]
    ) -> pd.DataFrame:
        base = self._validate_base_frame(dataframe)
        generated = {spec["name"]: self._apply_spec(base, spec) for spec in specs}
        result = (
            pd.concat([base, pd.DataFrame(generated, index=base.index)], axis=1)
            if generated else base
        )
        expected = self.input_features_ + [spec["name"] for spec in specs]
        if list(result.columns) != expected:
            raise OntologySchemaError("Schema fold-local divergente do schema OWL congelado.")
        return result

    def _recalibrate_specs(
        self, train: pd.DataFrame, specs: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        recalibrated: List[Dict[str, Any]] = []
        for original in specs:
            spec = copy.deepcopy(original)
            if spec.get("kind") == "hierarchical_aggregate":
                sources = list(spec.get("sources") or [])
                values = train.loc[:, sources].apply(pd.to_numeric, errors="raise")
                if values.isna().any().any():
                    raise OntologySchemaError(
                        f"Valores ausentes ao recalibrar {spec.get('name')}."
                    )
                centers = values.mean(axis=0)
                scales = values.std(axis=0, ddof=0)
                safe_scales = scales.where(scales.abs() > 1e-12, 1.0)
                spec["centers"] = {name: float(centers[name]) for name in sources}
                spec["scales"] = {name: float(safe_scales[name]) for name in sources}
                spec["fit_scope"] = "inner_fold_training_only"
            recalibrated.append(spec)
        return recalibrated

    def _entity_by_name(self) -> Dict[str, Any]:
        entities = ontology_entities(self.ontology)
        all_entities = sum(entities.values(), [])
        return {getattr(entity, "name", ""): entity for entity in all_entities}

    @staticmethod
    def _numeric(series: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(series) or pd.to_numeric(
            series, errors="coerce"
        ).notna().mean() >= 0.95

    def _fit_hierarchical_specs(self, train: pd.DataFrame) -> List[Dict[str, Any]]:
        by_name = self._entity_by_name()
        inferred_parents: set = set()
        parent_columns: Dict[str, List[str]] = defaultdict(list)
        for column, entity_name in self.column_entity_map_.items():
            entity = by_name.get(entity_name)
            if entity is None or not self._numeric(train[column]):
                continue
            parents = []
            if entity_kind(entity) in ("datatype_property", "object_property"):
                try:
                    parents.extend(getattr(item, "name", str(item)) for item in entity.domain)
                except Exception:
                    pass
            try:
                parents.extend(
                    getattr(item, "name", str(item)) for item in getattr(entity, "is_a", [])
                )
            except Exception:
                pass
            for parent in parents:
                if parent and parent not in ("Thing", "DatatypeProperty", "ObjectProperty"):
                    parent_columns[parent].append(column)
                    if (entity_name, parent) in self._inferred_subclass_pairs:
                        inferred_parents.add(parent)
        specs = []
        for parent, columns in parent_columns.items():
            sources = list(dict.fromkeys(columns))
            if len(sources) < 2:
                continue
            values = train.loc[:, sources].apply(pd.to_numeric, errors="raise")
            if values.isna().any().any():
                continue
            centers = values.mean(axis=0)
            scales = values.std(axis=0, ddof=0)
            safe_scales = scales.where(scales.abs() > 1e-12, 1.0)
            specs.append({
                "kind": "hierarchical_aggregate",
                "name": f"{self.ONTO_PREFIX}{parent}_aggregate",
                "sources": sources,
                "reasoner_inferred": parent in inferred_parents,
                "method": "standardized_mean",
                "centers": {name: float(centers[name]) for name in sources},
                "scales": {name: float(safe_scales[name]) for name in sources},
                "fit_scope": "training_only",
                "provenance": "owl_hierarchy_standardized_train_only",
            })
        return specs

    @staticmethod
    def _annotation_values(entity: Any, names: Sequence[str]) -> List[str]:
        """Valores de anotação por propriedade/IRI (ver :func:`core.ontology_quality.annotation_values`)."""
        return annotation_values(entity, names)

    @staticmethod
    def _safe_token(value: str) -> str:
        token = re.sub(r"[^0-9A-Za-z_]+", "_", str(value).strip())
        token = re.sub(r"_+", "_", token).strip("_")
        return token or "semantic_family"

    @staticmethod
    def _canonical_role(value: str) -> Optional[str]:
        token = re.sub(r"[^a-z0-9]+", "", str(value).lower())
        aliases = {
            "mean": "mean", "average": "mean", "avg": "mean",
            "worst": "worst", "maximum": "worst", "max": "worst", "extreme": "worst",
            "error": "error", "standarderror": "error", "stderr": "error", "se": "error",
        }
        return aliases.get(token)

    def _entity_family_role(self, entity: Any) -> Tuple[Optional[str], Optional[str]]:
        families = self._annotation_values(
            entity,
            ("measurementFamily", "semanticFamily", "featureFamily", "measurement_family"),
        )
        roles = self._annotation_values(
            entity,
            ("statisticRole", "measurementRole", "semanticRole", "statistic_role"),
        )
        family = families[0] if families else None
        role = next((self._canonical_role(item) for item in roles if self._canonical_role(item)), None)
        return family, role

    def _fit_relational_specs(self, train: pd.DataFrame) -> List[Dict[str, Any]]:
        """Cria relações dimensionais apenas quando a OWL declara família+papel.

        Não há regras hardcoded por dataset. Uma ontologia pode declarar, por
        exemplo, três propriedades da mesma ``measurementFamily`` com papéis
        ``mean``, ``error`` e ``worst``. A partir disso são gerados contrastes
        semânticos auditáveis e independentes de rótulos.
        """
        by_name = self._entity_by_name()
        families: Dict[str, Dict[str, str]] = defaultdict(dict)
        family_entities: Dict[str, Dict[str, str]] = defaultdict(dict)
        for column, entity_name in self.column_entity_map_.items():
            if column not in train.columns or not self._numeric(train[column]):
                continue
            entity = by_name.get(entity_name)
            if entity is None:
                continue
            family, role = self._entity_family_role(entity)
            if not family or not role:
                continue
            families[str(family)][role] = column
            family_entities[str(family)][role] = entity_name

        specs: List[Dict[str, Any]] = []
        for family, roles in sorted(families.items(), key=lambda item: str(item[0])):
            token = self._safe_token(family)
            entities = family_entities[family]
            if "mean" in roles and "worst" in roles:
                specs.append({
                    "kind": "relational",
                    "name": f"{self.ONTO_PREFIX}{token}_worst_minus_mean",
                    "operation": "difference",
                    "left": roles["worst"],
                    "right": roles["mean"],
                    "sources": [roles["worst"], roles["mean"]],
                    "family": family,
                    "roles": ["worst", "mean"],
                    "owl_entities": [entities["worst"], entities["mean"]],
                    "provenance": "owl_measurement_family_role",
                })
                specs.append({
                    "kind": "relational",
                    "name": f"{self.ONTO_PREFIX}{token}_relative_worst_delta",
                    "operation": "relative_delta",
                    "left": roles["worst"],
                    "right": roles["mean"],
                    "denominator": roles["mean"],
                    "sources": [roles["worst"], roles["mean"]],
                    "family": family,
                    "roles": ["worst", "mean"],
                    "owl_entities": [entities["worst"], entities["mean"]],
                    "provenance": "owl_measurement_family_role",
                })
            if "mean" in roles and "worst" in roles:
                specs.append({
                    "kind": "relational",
                    "name": f"{self.ONTO_PREFIX}{token}_family_contrast",
                    "operation": "contrast",
                    "left": roles["worst"],
                    "right": roles["mean"],
                    "sources": [roles["worst"], roles["mean"]],
                    "family": family,
                    "roles": ["worst", "mean"],
                    "owl_entities": [entities["worst"], entities["mean"]],
                    "provenance": "owl_measurement_family_role",
                })
            if "mean" in roles and "worst" in roles and "error" in roles:
                specs.append({
                    "kind": "relational",
                    "name": f"{self.ONTO_PREFIX}{token}_normalized_error",
                    "operation": "normalized_error",
                    "left": roles["error"],
                    "right": roles["mean"],
                    "denominator": roles["worst"],
                    "sources": [roles["error"], roles["mean"], roles["worst"]],
                    "family": family,
                    "roles": ["error", "mean", "worst"],
                    "owl_entities": [entities["error"], entities["mean"], entities["worst"]],
                    "provenance": "owl_measurement_family_role",
                })
            if "mean" in roles and "error" in roles:
                specs.append({
                    "kind": "relational",
                    "name": f"{self.ONTO_PREFIX}{token}_error_ratio",
                    "operation": "ratio",
                    "left": roles["error"],
                    "denominator": roles["mean"],
                    "sources": [roles["error"], roles["mean"]],
                    "family": family,
                    "roles": ["error", "mean"],
                    "owl_entities": [entities["error"], entities["mean"]],
                    "provenance": "owl_measurement_family_role",
                })
        return specs

    def _explicit_bounds(self, entity) -> Dict[str, Optional[float]]:
        bounds = {"low": None, "high": None}
        if entity is None:
            return bounds
        entity_name = getattr(entity, "name", "")
        if self.matcher is not None and hasattr(self.matcher, "datatype_ranges"):
            info = self.matcher.datatype_ranges.get(entity_name, {}) or {}
            for key, source in (("low", "min"), ("high", "max")):
                if info.get(source) is not None:
                    bounds[key] = float(info[source])
        for attr, key in (
            ("minInclusive", "low"), ("min_inclusive", "low"),
            ("maxInclusive", "high"), ("max_inclusive", "high"),
        ):
            value = getattr(entity, attr, None)
            if value is not None:
                try:
                    bounds[key] = float(value[0] if isinstance(value, list) else value)
                except (TypeError, ValueError):
                    pass
        # Formato OWL padrão: facetas xsd:min/maxInclusive num rdfs:range
        # (ConstrainedDatatype), já lidas pelo grafo semântico.
        try:
            low, high = _explicit_entity_bounds(entity)
        except Exception:
            low = high = None
        if bounds["low"] is None and low is not None:
            bounds["low"] = float(low)
        if bounds["high"] is None and high is not None:
            bounds["high"] = float(high)
        return bounds

    def _fit_constraint_specs(self, train: pd.DataFrame) -> List[Dict[str, Any]]:
        by_name = self._entity_by_name()
        specs = []
        for column, entity_name in self.column_entity_map_.items():
            if not self._numeric(train[column]):
                continue
            series = pd.to_numeric(train[column], errors="coerce")
            if series.isna().any():
                continue
            bounds = self._explicit_bounds(by_name.get(entity_name))
            provenance = "owl_explicit_bound"
            if bounds["low"] is None and bounds["high"] is None:
                if not self.allow_train_calibrated_bounds:
                    continue
                bounds = {
                    "low": float(series.quantile(0.25)),
                    "high": float(series.quantile(0.75)),
                }
                provenance = "training_quantile_not_owl"
            if bounds["low"] is not None:
                specs.append({
                    "kind": "constraint", "operator": "ge",
                    "name": f"{self.ONTO_PREFIX}{entity_name}_within_min",
                    "source": column, "threshold": float(bounds["low"]),
                    "provenance": provenance,
                })
            if bounds["high"] is not None:
                specs.append({
                    "kind": "constraint", "operator": "le",
                    "name": f"{self.ONTO_PREFIX}{entity_name}_within_max",
                    "source": column, "threshold": float(bounds["high"]),
                    "provenance": provenance,
                })
        return specs

    @staticmethod
    def _semantic_group(entity) -> Optional[str]:
        candidates = []
        for attr in ("is_a", "is_instance_of"):
            try:
                candidates.extend(getattr(entity, attr, []) or [])
            except Exception:
                pass
        excluded = {"Thing", "NamedIndividual", "owl.Thing"}
        names = [getattr(item, "name", None) for item in candidates]
        return next((name for name in names if name and name not in excluded), None)

    def _fit_categorical_specs(self, train: pd.DataFrame) -> List[Dict[str, Any]]:
        entities = ontology_entities(self.ontology)
        value_entities = entities["individuals"] + entities["classes"]
        specs = []
        for column in self.input_features_:
            if self._numeric(train[column]):
                continue
            values = [str(value) for value in pd.Series(train[column]).dropna().unique()]
            mapping, audit = {}, {}
            for value in values:
                ranked = sorted(
                    ((self.quality_gate._candidate_score(value, entity), entity) for entity in value_entities),
                    key=lambda pair: pair[0], reverse=True,
                )
                score, entity = ranked[0] if ranked else (0.0, None)
                runner = ranked[1][0] if len(ranked) > 1 else 0.0
                accepted = bool(
                    entity is not None and score >= self.quality_gate.min_match_score
                    and (score >= 0.98 or score - runner >= self.quality_gate.min_ambiguity_margin)
                )
                group = self._semantic_group(entity) if accepted else None
                audit[value] = {
                    "entity": getattr(entity, "name", None) if accepted else None,
                    "entity_type": entity_kind(entity) if accepted else None,
                    "score": float(score), "group": group,
                }
                if accepted and group:
                    mapping[value] = group
            groups = list(dict.fromkeys(mapping.values()))
            self.category_matching_[column] = audit
            if len(mapping) == len(values) and 1 < len(groups) < len(values):
                codes = {group: index + 1 for index, group in enumerate(groups)}
                specs.append({
                    "kind": "categorical_group",
                    "name": f"{self.ONTO_PREFIX}{column}_semantic_group",
                    "source": column,
                    "mapping": {value: codes[group] for value, group in mapping.items()},
                    "group_labels": codes,
                    "provenance": "owl_named_individual_type",
                })
        return specs

    def _deduplicate_specs(
        self, train: pd.DataFrame, specs: Sequence[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        accepted, existing = [], [train[column] for column in train.columns]
        names = set(train.columns)
        existing_names = list(train.columns)
        self.feature_audit_ = []

        def describe_rule(spec):
            if spec.get("kind") == "hierarchical_aggregate":
                return (
                    "standardized_mean(" + ", ".join(map(str, spec.get("sources", []))) + ")"
                )
            if spec.get("kind") == "relational":
                op = spec.get("operation")
                if op == "difference":
                    return f"{spec.get('left')} - {spec.get('right')}"
                if op == "relative_delta":
                    return (
                        f"({spec.get('left')} - {spec.get('right')}) / "
                        f"abs({spec.get('denominator')})"
                    )
                if op == "ratio":
                    return f"{spec.get('left')} / abs({spec.get('denominator')})"
                if op == "contrast":
                    return (
                        f"({spec.get('left')} - {spec.get('right')}) / "
                        f"(abs({spec.get('left')}) + abs({spec.get('right')}))"
                    )
                if op == "normalized_error":
                    return (
                        f"abs({spec.get('left')}) / "
                        f"(abs({spec.get('right')}) + abs({spec.get('denominator')}))"
                    )
                return f"relational:{op}"
            if spec.get("kind") == "constraint":
                op = ">=" if spec.get("operator") == "ge" else "<="
                return f"{spec.get('source')} {op} {spec.get('threshold')}"
            if spec.get("kind") == "categorical_group":
                return f"OWL semantic group mapping({spec.get('source')})"
            return str(spec.get("kind", "unknown"))

        def owl_origin(spec):
            sources = list(spec.get("sources") or [])
            if spec.get("source") is not None:
                sources.append(spec.get("source"))
            entities = [
                self.column_entity_map_.get(str(source))
                for source in sources
                if self.column_entity_map_.get(str(source))
            ]
            entities.extend(spec.get("owl_entities") or [])
            return {
                "source_columns": [str(source) for source in sources],
                "owl_entities_or_properties": list(dict.fromkeys(map(str, entities))),
                "provenance": spec.get("provenance"),
            }

        for spec in specs:
            audit_row = {
                "feature": str(spec.get("name")),
                "kind": str(spec.get("kind")),
                "owl_origin": owl_origin(spec),
                "derivation_rule": describe_rule(spec),
                "provenance": spec.get("provenance"),
                "accepted": False,
                "reason_not_duplicated": None,
                "rejection_reason": None,
                "duplicate_of": None,
                "constant_rate": None,
                "effect_on_metrics": None,
                "fold_stability": None,
                "linear_redundancy_r2": None,
                # OWL = conhecimento declarado na ontologia; statistical = limiar/valor
                # aprendido nos dados de treino (não pode ser apresentado como OWL).
                "knowledge_source": (
                    "ontology" if str(spec.get("provenance") or "").startswith("owl")
                    else "statistical"
                ),
                "reasoner_inferred": bool(spec.get("reasoner_inferred", False)),
            }
            if spec["name"] in names:
                audit_row["rejection_reason"] = "name_collision"
                self.feature_audit_.append(audit_row)
                continue
            try:
                series = self._apply_spec(train, spec)
            except Exception as exc:
                audit_row["rejection_reason"] = f"derivation_error:{type(exc).__name__}"
                self.feature_audit_.append(audit_row)
                continue
            value_counts = pd.Series(series).value_counts(dropna=False, normalize=True)
            audit_row["constant_rate"] = float(value_counts.max()) if len(value_counts) else 1.0
            if series.nunique(dropna=False) <= 1:
                audit_row["rejection_reason"] = "constant"
                self.feature_audit_.append(audit_row)
                continue
            if audit_row["constant_rate"] >= self.low_variance_rate:
                audit_row["rejection_reason"] = "low_variance"
                self.feature_audit_.append(audit_row)
                continue
            duplicate = False
            duplicate_of = None
            for other_name, other in zip(existing_names, existing):
                a = pd.Series(series).reset_index(drop=True)
                b = pd.Series(other).reset_index(drop=True)
                if a.equals(b):
                    duplicate = True
                    duplicate_of = str(other_name)
                    break
                try:
                    if np.allclose(
                        pd.to_numeric(a, errors="raise"),
                        pd.to_numeric(b, errors="raise"),
                        equal_nan=True,
                    ):
                        duplicate = True
                        duplicate_of = str(other_name)
                        break
                except Exception:
                    pass
            if not duplicate:
                near = self._near_duplicate_of(series, existing_names, existing)
                if near is not None:
                    audit_row["rejection_reason"] = "near_duplicate"
                    audit_row["duplicate_of"] = near
                    self.feature_audit_.append(audit_row)
                    continue
            if not duplicate:
                r2 = self._linear_redundancy(train, spec, series)
                audit_row["linear_redundancy_r2"] = r2
                if (
                    self.drop_linear_redundant
                    and r2 is not None
                    and r2 >= self.linear_redundancy_threshold
                ):
                    audit_row["rejection_reason"] = "linear_combination_of_sources"
                    self.feature_audit_.append(audit_row)
                    continue
                accepted.append({**spec, "knowledge_source": audit_row["knowledge_source"]})
                existing.append(series)
                existing_names.append(spec["name"])
                names.add(spec["name"])
                audit_row["accepted"] = True
                audit_row["reason_not_duplicated"] = "distinct_from_original_and_prior_derived_features"
            else:
                audit_row["rejection_reason"] = "deterministic_duplicate"
                audit_row["duplicate_of"] = duplicate_of
            self.feature_audit_.append(audit_row)
        reasons = defaultdict(int)
        for row in self.feature_audit_:
            if row.get("rejection_reason"):
                reasons[row["rejection_reason"]] += 1
        self.generation_summary_ = {
            "generated": len(specs),
            "removed_constant": reasons["constant"],
            "removed_low_variance": reasons["low_variance"],
            "removed_duplicate": reasons["deterministic_duplicate"],
            "removed_near_duplicate": reasons["near_duplicate"],
            "removed_linear_redundant": reasons["linear_combination_of_sources"],
            "removed_name_collision": reasons["name_collision"],
            "removed_derivation_error": sum(
                n for k, n in reasons.items() if k.startswith("derivation_error")
            ),
            "retained": len(accepted),
            "retained_ontology_knowledge": sum(
                1 for r in self.feature_audit_ if r["accepted"] and r["knowledge_source"] == "ontology"
            ),
            "retained_statistical": sum(
                1 for r in self.feature_audit_ if r["accepted"] and r["knowledge_source"] == "statistical"
            ),
        }
        return accepted

    def _near_duplicate_of(self, series, existing_names, existing) -> Optional[str]:
        """Nome da feature existente com |correlação| ≥ limiar (quase duplicada)."""
        try:
            a = pd.to_numeric(pd.Series(series).reset_index(drop=True), errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(a).all() or a.std() <= 0:
            return None
        for name, other in zip(existing_names, existing):
            try:
                b = pd.to_numeric(pd.Series(other).reset_index(drop=True), errors="raise").to_numpy(dtype=float)
            except (TypeError, ValueError):
                continue
            if len(b) != len(a) or not np.isfinite(b).all() or b.std() <= 0:
                continue
            corr = abs(float(np.corrcoef(a, b)[0, 1]))
            if np.isfinite(corr) and corr >= self.near_duplicate_correlation:
                return str(name)
        return None

    def _linear_redundancy(
        self, train: pd.DataFrame, spec: Dict[str, Any], series: pd.Series
    ) -> Optional[float]:
        """R² da feature derivada contra as suas fontes (regressão linear com intercepto).

        Só faz sentido para agregados e relações numéricas; restrições e grupos
        categóricos são não lineares por natureza e devolvem ``None``. Usa apenas
        o treino. R² ≈ 1 significa que a feature não traz informação nova a um
        modelo linear.
        """
        if spec.get("kind") not in {"hierarchical_aggregate", "relational"}:
            return None
        sources = [str(c) for c in (spec.get("sources") or []) if c in train.columns]
        if not sources:
            return None
        try:
            design = train[sources].apply(pd.to_numeric, errors="raise").to_numpy(dtype=float)
            target = pd.to_numeric(pd.Series(series).reset_index(drop=True), errors="raise").to_numpy(dtype=float)
        except (TypeError, ValueError):
            return None
        keep = np.isfinite(design).all(axis=1) & np.isfinite(target)
        if keep.sum() <= len(sources) + 1:
            return None
        design, target = design[keep], target[keep]
        total = float(((target - target.mean()) ** 2).sum())
        if total <= 0.0:
            return None
        coef, *_ = np.linalg.lstsq(
            np.column_stack([np.ones(len(design)), design]), target, rcond=None
        )
        residual = target - np.column_stack([np.ones(len(design)), design]) @ coef
        return float(max(0.0, 1.0 - float((residual ** 2).sum()) / total))

    def _apply_spec(self, frame: pd.DataFrame, spec: Dict[str, Any]) -> pd.Series:
        kind = spec["kind"]
        if kind == "hierarchical_aggregate":
            values = frame.loc[:, spec["sources"]].apply(pd.to_numeric, errors="raise")
            if values.isna().any().any():
                raise OntologySchemaError(
                    f"Valores ausentes nas fontes de {spec['name']}; imputação explícita necessária."
                )
            if spec.get("method") == "standardized_mean":
                centers = spec.get("centers") or {}
                scales = spec.get("scales") or {}
                normalized = values.copy()
                for source in spec["sources"]:
                    center = float(centers.get(source, 0.0))
                    scale = float(scales.get(source, 1.0))
                    if not np.isfinite(scale) or abs(scale) <= 1e-12:
                        scale = 1.0
                    normalized[source] = (values[source] - center) / scale
                return normalized.mean(axis=1)
            return values.mean(axis=1)
        if kind == "relational":
            left = pd.to_numeric(frame[spec["left"]], errors="raise")
            right_name = spec.get("right")
            right = (
                pd.to_numeric(frame[right_name], errors="raise")
                if right_name is not None else None
            )
            denominator_name = spec.get("denominator")
            denominator = (
                pd.to_numeric(frame[denominator_name], errors="raise")
                if denominator_name is not None else None
            )
            series_to_check = [left]
            if right is not None:
                series_to_check.append(right)
            if denominator is not None:
                series_to_check.append(denominator)
            if any(series.isna().any() for series in series_to_check):
                raise OntologySchemaError(
                    f"Valores ausentes nas fontes relacionais de {spec['name']}."
                )
            operation = spec.get("operation")
            if operation == "difference":
                return left - right
            if operation == "relative_delta":
                denom = denominator.abs().clip(lower=1e-12)
                return (left - right) / denom
            if operation == "ratio":
                denom = denominator.abs().clip(lower=1e-12)
                return left / denom
            if operation == "contrast":
                return (left - right) / (left.abs() + right.abs()).clip(lower=1e-12)
            if operation == "normalized_error":
                return left.abs() / (right.abs() + denominator.abs()).clip(lower=1e-12)
            raise OntologySchemaError(
                f"Operação relacional desconhecida em {spec['name']}: {operation}"
            )
        if kind == "constraint":
            values = pd.to_numeric(frame[spec["source"]], errors="raise")
            if values.isna().any():
                raise OntologySchemaError(
                    f"Valores ausentes em {spec['source']}; não serão preenchidos com zero."
                )
            if spec["operator"] == "ge":
                return (values >= spec["threshold"]).astype(int)
            return (values <= spec["threshold"]).astype(int)
        if kind == "categorical_group":
            raw = frame[spec["source"]].astype(str)
            unknown = sorted(set(raw) - set(spec["mapping"]))
            if unknown:
                raise OntologySchemaError(
                    f"Categorias sem mapping OWL em {spec['source']}: {unknown}."
                )
            return raw.map(spec["mapping"]).astype(int)
        raise OntologySchemaError(f"Spec ontológica desconhecida: {kind}")

    def prune_semantically_incoherent_rules(
        self, rules_text: str, feature_names: List[str],
        feature_semantics: Optional[Dict[int, Dict]] = None,
        validation_result: Optional[Dict] = None,
    ) -> Tuple[str, List[str]]:
        if not rules_text:
            return rules_text, []
        issue_features = {
            name for issue in (validation_result or {}).get("issues", [])
            for name in feature_names if name in str(issue)
        }
        flagged, output = [], []
        for line in rules_text.splitlines():
            if any(name in line for name in issue_features):
                flagged.append(line.strip())
                output.append(f"{line}  [incoerente segundo o reasoner]")
            else:
                output.append(line)
        return "\n".join(output), flagged

    def log_feature_engineering_summary(self, stats: Optional[Dict] = None):
        values = stats or self.last_engineering_stats
        if not values:
            return
        print(f"[ONTOLOGY FIT] escopo={values.get('fit_scope')} linhas={values.get('fit_rows')}")
        print(
            f"[ONTOLOGY FIT] originais={values.get('original_features', 0)} "
            f"mapeadas={values.get('mapped_direct', 0)} "
            f"derivadas={len(values.get('inferred_columns', []))} "
            f"total={values.get('total_training_features', 0)}"
        )
