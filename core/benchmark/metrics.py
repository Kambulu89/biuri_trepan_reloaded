"""Módulo ÚNICO de métricas do benchmark.

Duas famílias que NUNCA se misturam:
* ``*_real_labels``  -> modelo vs rótulos reais (desempenho preditivo);
* ``fidelity_to_oracle`` -> surrogate vs predições do SEU oráculo (e o nome do oráculo é obrigatório).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score,
)

REAL = "real_labels"


@dataclass(frozen=True)
class OracleInfo:
    """Identificação obrigatória do professor de um surrogate."""

    name: str
    type: str
    feature_space: int
    version: str = "unversioned"
    accuracy_real_labels: Optional[float] = None

    def __post_init__(self):
        if not self.name or not str(self.name).strip():
            raise ValueError("OracleInfo.name é obrigatório: fidelity sem oráculo identificado é inválida.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _arr(v) -> np.ndarray:
    return np.asarray(v)


def accuracy_real_labels(y_real, prediction) -> float:
    """Accuracy contra os RÓTULOS REAIS (nunca chamar Fidelity)."""
    y_real, prediction = _arr(y_real), _arr(prediction)
    _check(y_real, prediction)
    return float(accuracy_score(y_real, prediction))


def fidelity_to_oracle(oracle_prediction, surrogate_prediction, oracle: OracleInfo) -> float:
    """Concordância surrogate vs oráculo: accuracy_score(oracle_pred, surrogate_pred).

    ``oracle`` é obrigatório e o valor só tem significado relativamente a ele."""
    if not isinstance(oracle, OracleInfo):
        raise TypeError("fidelity_to_oracle exige um OracleInfo (oráculo explícito).")
    o, s = _arr(oracle_prediction), _arr(surrogate_prediction)
    _check(o, s)
    return float(accuracy_score(o, s))


def format_fidelity(value: float, surrogate: str, oracle: OracleInfo) -> str:
    """Nunca mostra apenas 'Fidelity = 0.94': inclui sempre o oráculo."""
    return f"Fidelity {surrogate} -> {oracle.name}: {value:.4f}"


def minority_class(y_train, labels: Optional[Sequence] = None):
    """Classe minoritária determinada SÓ no treino (empate: primeira na ordem de ``labels``)."""
    y = _arr(y_train)
    labs = list(labels) if labels is not None else list(np.unique(y))
    counts = [(int(np.sum(y == l)), i, l) for i, l in enumerate(labs)]
    counts = [c for c in counts if c[0] > 0] or counts
    return min(counts, key=lambda c: (c[0], c[1]))[2]


def classification_bundle(y_true, y_pred, *, labels: Optional[Sequence] = None, minority_label=None,
                          suffix: str = REAL) -> Dict[str, float]:
    """Métricas de classificação binária/multiclasse, com média macro explícita e ordem de classes fixa."""
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    _check(y_true, y_pred)
    labs = list(labels) if labels is not None else list(np.unique(np.concatenate([y_true, y_pred])))
    out = {
        f"accuracy_{suffix}": float(accuracy_score(y_true, y_pred)),
        f"balanced_accuracy_{suffix}": float(balanced_accuracy_score(y_true, y_pred)),
        f"precision_macro_{suffix}": float(precision_score(y_true, y_pred, labels=labs, average="macro", zero_division=0)),
        f"recall_macro_{suffix}": float(recall_score(y_true, y_pred, labels=labs, average="macro", zero_division=0)),
        f"macro_f1_{suffix}": float(f1_score(y_true, y_pred, labels=labs, average="macro", zero_division=0)),
        f"weighted_f1_{suffix}": float(f1_score(y_true, y_pred, labels=labs, average="weighted", zero_division=0)),
    }
    if minority_label is not None:
        mask = y_true == minority_label
        out[f"minority_recall_{suffix}"] = float(np.mean(y_pred[mask] == minority_label)) if mask.any() else float("nan")
    return out


def evaluate_model(*, y_real, prediction, labels: Sequence, minority_label=None,
                   oracle: Optional[OracleInfo] = None, oracle_prediction=None) -> Dict[str, Any]:
    """Avaliação completa de UM modelo. Fidelity só existe quando há oráculo (surrogates)."""
    out: Dict[str, Any] = classification_bundle(y_real, prediction, labels=labels, minority_label=minority_label)
    if oracle is not None:
        if oracle_prediction is None:
            raise ValueError("Fidelity exige as predições do oráculo no mesmo conjunto de avaliação.")
        out["fidelity_to_oracle"] = fidelity_to_oracle(oracle_prediction, prediction, oracle)
        out["fidelity_oracle_name"] = oracle.name
        out["fidelity_oracle_type"] = oracle.type
    else:
        out["fidelity_to_oracle"] = None
        out["fidelity_oracle_name"] = None
        out["fidelity_oracle_type"] = None
    return out


def _check(a: np.ndarray, b: np.ndarray) -> None:
    if a.shape[0] != b.shape[0] or a.shape[0] == 0:
        raise ValueError("Vectores de comprimento diferente ou vazios.")


FIDELITY_FIELDS = ("fidelity_to_oracle", "fidelity_oracle_name", "fidelity_oracle_type")
PREDICTIVE_FIELDS = tuple(f"{m}_{REAL}" for m in (
    "accuracy", "balanced_accuracy", "precision_macro", "recall_macro", "macro_f1", "weighted_f1", "minority_recall"))
