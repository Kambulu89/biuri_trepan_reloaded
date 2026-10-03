"""Objecto de resultado padrão e estados explícitos (nunca ``None`` sem explicação)."""
from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


class CounterfactualStatus(str, enum.Enum):
    SUCCESS = "SUCCESS"
    NO_COUNTERFACTUAL_FOUND = "NO_COUNTERFACTUAL_FOUND"
    INVALID_TARGET = "INVALID_TARGET"
    UNSUPPORTED_FEATURE_SPACE = "UNSUPPORTED_FEATURE_SPACE"
    CONSTRAINT_INFEASIBLE = "CONSTRAINT_INFEASIBLE"
    METHOD_FAILURE = "METHOD_FAILURE"
    TIMEOUT = "TIMEOUT"


def _py(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _py(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_py(v) for v in value]
    return value


@dataclass
class CounterfactualCandidate:
    """Um CF já re-validado no modelo explicado."""

    vector: List[float]
    human: Dict[str, Any]                  # espaço compreensível (one-hot reconstruído)
    changes: List[Dict[str, Any]]          # só alterações, ao nível humano
    predicted_class: Any
    model_valid: bool                      # model.predict(cf) == target (reavaliado)
    semantic_valid: Optional[bool]         # None = NOT_AVAILABLE
    semantic_status: str
    hard_violations: List[str]
    soft_warnings: List[str]
    proximity: float                       # distância Gower (menor = mais próximo)
    sparsity: int                          # nº de features HUMANAS alteradas
    plausibility: Optional[float]
    plausible: Optional[bool]
    original_probability: Optional[float] = None       # P(classe original | x)
    counterfactual_probability: Optional[float] = None  # P(classe alvo | cf)
    original_target_probability: Optional[float] = None  # P(classe alvo | x), para ver o ganho de confiança
    cost_weighted_distance: Optional[float] = None
    method: str = ""
    rule: Optional[str] = None
    dominated_by: List[int] = field(default_factory=list)
    cross_model: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def fully_valid(self) -> bool:
        return bool(self.model_valid and self.semantic_valid is not False and not self.hard_violations)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["fully_valid"] = self.fully_valid
        return _py(d)


@dataclass
class CounterfactualResult:
    method: str
    status: CounterfactualStatus
    model_explained: str
    dataset: str
    original_instance: List[float]
    original_human: Dict[str, Any]
    original_class: Any
    target_class: Any
    predicted_class: Any                      # classe prevista para o melhor CF (ou None)
    candidates: List[CounterfactualCandidate] = field(default_factory=list)
    changed_features: List[str] = field(default_factory=list)
    proximity: Optional[float] = None
    sparsity: Optional[int] = None
    plausibility: Optional[float] = None
    semantic_validation: str = "NOT_AVAILABLE"
    runtime: float = 0.0
    seed: Optional[int] = None
    warnings: List[str] = field(default_factory=list)
    message: str = ""                          # explicação sempre presente (sobretudo se não for SUCCESS)
    instance_id: Any = None
    iterations: int = 0
    terminated_by: str = "completed"           # completed | max_iterations | max_time | found_enough
    diversity: Dict[str, Optional[float]] = field(default_factory=dict)
    rejected: List[Dict[str, Any]] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    provenance: Dict[str, Any] = field(default_factory=dict)
    explanation: str = ""
    canonical: bool = False
    method_label: str = ""

    @property
    def best(self) -> Optional[CounterfactualCandidate]:
        return self.candidates[0] if self.candidates else None

    @property
    def counterfactual(self) -> Optional[List[float]]:
        return self.best.vector if self.best else None

    def to_dict(self, include_rejected: bool = True) -> Dict[str, Any]:
        d = {k: v for k, v in asdict(self).items() if k not in ("candidates", "rejected")}
        d["status"] = self.status.value
        d["candidates"] = [c.to_dict() for c in self.candidates]
        d["counterfactual"] = self.counterfactual
        if include_rejected:
            d["rejected"] = _py(self.rejected)
        return _py(d)
