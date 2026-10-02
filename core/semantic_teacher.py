"""Professor MLP+OWL para o TREPAN (oráculo no espaço do modelo).

O TREPAN consulta o oráculo com matrizes no espaço do modelo (``Z``, saída do
``DataPreprocessor``). O professor ontológico reconstrói, a partir de ``Z``, as colunas
originais numéricas, aplica o ``OntologyProcessor`` **já ajustado no treino**, acrescenta só
as features semânticas selecionadas e consulta o MLP treinado nesse espaço. O TREPAN
Original e o Reloaded continuam a usar o MESMO objeto professor: a única diferença entre
eles permanece a semântica de seleção de splits.

Só é usado quando:

1. o enriquecimento foi aceite (``semantic_mlp_accepted``) com evidência pelo menos
   ``min_evidence`` (por omissão ``strong``: IC da utilidade acima de zero);
2. todas as entradas do processador são colunas numéricas não transformadas em ``Z``
   (caso contrário não é possível reconstruí-las; fica ``UNAVAILABLE`` com a razão).

Nenhum dado de teste participa: o MLP é afinado e ajustado no treino de desenvolvimento.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from core.semantic_enrichment import (
    EnrichmentConfig, SemanticEnrichmentResult, _mlp_pipeline, _tune,
)

EVIDENCE_ORDER = {"n/a": 0, "weak": 1, "strong": 2}


@dataclass
class TeacherDecision:
    use: bool
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)


def decide_semantic_teacher(report: Optional[Dict[str, Any]], min_evidence: Optional[str] = "strong") -> TeacherDecision:
    """Política de seleção (sem tocar no teste): aceite + evidência suficiente."""
    if not report:
        return TeacherDecision(False, "no_enrichment_report")
    if min_evidence is None:
        return TeacherDecision(False, "semantic_teacher_disabled")
    if min_evidence not in EVIDENCE_ORDER or min_evidence == "n/a":
        raise ValueError("min_evidence deve ser 'strong' ou 'weak'.")
    if not report.get("semantic_mlp_accepted"):
        return TeacherDecision(False, f"enrichment_not_accepted:{report.get('decision')}")
    strength = report.get("evidence_strength", "n/a")
    if EVIDENCE_ORDER.get(strength, 0) < EVIDENCE_ORDER[min_evidence]:
        return TeacherDecision(False, f"evidence_{strength}_below_required_{min_evidence}",
                               {"evidence_strength": strength})
    if not report.get("selected_semantic_features"):
        return TeacherDecision(False, "no_selected_semantic_features")
    return TeacherDecision(True, "accepted_with_sufficient_evidence", {"evidence_strength": strength})


def teacher_inputs_available(processor, z_feature_names: Sequence[str]) -> TeacherDecision:
    """O professor só é reconstruível se cada entrada do processador for uma coluna de ``Z``."""
    names = [str(n) for n in z_feature_names]
    missing = [c for c in processor.input_features_ if c not in names]
    if missing:
        return TeacherDecision(False, "inputs_not_reconstructable_from_model_space",
                               {"missing_in_model_space": missing[:10]})
    return TeacherDecision(True, "inputs_reconstructable")


def detach_ontology(processor):
    """Cópia do processador sem o objeto owlready2 (não serializável); o transform só usa as specs."""
    clone = copy.copy(processor)
    clone.ontology = None
    clone.matcher = None
    return clone


class SemanticTeacher:
    """Oráculo MLP+OWL compatível com o contrato ``predict``/``predict_proba``/``classes_``."""

    ORACLE_TYPE = "semantic_mlp_teacher"

    def __init__(self, processor, input_columns: Sequence[str], z_indices: Sequence[int],
                 selected_features: Sequence[str], pipeline, audit: Dict[str, Any]):
        self.processor = processor
        self.input_columns = [str(c) for c in input_columns]
        self.z_indices = [int(i) for i in z_indices]
        self.selected_features = [str(f) for f in selected_features]
        self.pipeline = pipeline
        self.classes_ = pipeline.classes_
        self.audit_ = dict(audit)

    def _enriched(self, Z) -> pd.DataFrame:
        Z = np.asarray(Z, dtype=float)
        if Z.ndim != 2 or max(self.z_indices) >= Z.shape[1]:
            raise ValueError("Matriz Z incompatível com o professor semântico.")
        frame = pd.DataFrame(Z[:, self.z_indices], columns=self.input_columns)
        full = self.processor.transform(frame)
        return full[self.input_columns + self.selected_features]

    def predict(self, Z):
        return self.pipeline.predict(self._enriched(Z))

    def predict_proba(self, Z):
        return self.pipeline.predict_proba(self._enriched(Z))


def build_semantic_teacher(
    result: SemanticEnrichmentResult,
    Z_train,
    y_train,
    z_feature_names: Sequence[str],
    *,
    config: EnrichmentConfig,
    seed: int,
) -> SemanticTeacher:
    """Treina o MLP+OWL no treino de desenvolvimento, com o mesmo orçamento de otimização."""
    names = [str(n) for n in z_feature_names]
    availability = teacher_inputs_available(result.processor, names)
    if not availability.use:
        raise ValueError(f"Professor semântico indisponível: {availability.reason} {availability.details}")
    inputs = list(result.processor.input_features_)
    z_idx = [names.index(c) for c in inputs]
    Z_train = np.asarray(Z_train, dtype=float)
    frame = pd.DataFrame(Z_train[:, z_idx], columns=inputs)
    enriched = result.transform(frame)
    y = np.asarray(y_train)
    _, y_idx = np.unique(y, return_inverse=True)
    params = _tune(enriched, y_idx, config, seed)
    pipeline = _mlp_pipeline(enriched, params, config, seed)
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        pipeline.fit(enriched, y)
    audit = {
        "teacher": "mlp_semantic", "selected_semantic_features": list(result.selected_features),
        "n_input_features": len(inputs), "n_enriched_features": int(enriched.shape[1]),
        "hyperparameters": {k: (list(v) if isinstance(v, tuple) else v) for k, v in params.items()},
        "tuning_candidates": config.tuning_candidates, "trained_on": "development_training_only",
        "evidence_strength": result.report.get("evidence_strength"),
        "enrichment_decision": result.report.get("decision"),
    }
    return SemanticTeacher(detach_ontology(result.processor), inputs, z_idx,
                           result.selected_features, pipeline, audit)
