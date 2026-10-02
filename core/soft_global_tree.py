"""Compatibilidade de selecção global sem criar árvores auxiliares.

A build de produção V9.2 não permite treinar uma segunda família de árvore para
substituir ou redestilar TREPAN Original/Reloaded. Este módulo mantém apenas a
API de selecção de candidatos já treinados para compatibilidade histórica.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


@dataclass(frozen=True)
class GlobalTreeConfig:
    fidelity_weight: float = 0.35
    balanced_accuracy_weight: float = 0.20
    macro_f1_weight: float = 0.15
    minority_recall_weight: float = 0.10
    simplicity_weight: float = 0.20
    max_depths: tuple[int, ...] = (4, 6, 8)
    random_state: int = 42


class SoftDecisionTreeClassifier:
    """API retirada da produção.

    A antiga soft-tree pertencia a outra família algorítmica. Manter uma classe
    explícita que falha com mensagem accionável é preferível a um fallback
    silencioso ou a desserializar artefactos com semântica diferente.
    """
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def fit(self, *args, **kwargs):
        raise RuntimeError(
            "SoftDecisionTreeClassifier foi retirado da build de produção. "
            "Use TrepanOriginalClassifier ou TrepanReloadedClassifier."
        )


def select_global_surrogate(candidates: Dict[str, object], X_validation, y_validation, oracle, *, config: GlobalTreeConfig = GlobalTreeConfig()):
    """Selecciona entre modelos já treinados, sem criar/refitar árvores."""
    X = np.asarray(X_validation)
    y = np.asarray(y_validation)
    oracle_y = np.asarray(oracle.predict(X))
    rows = []
    valid = []
    for name, model in candidates.items():
        if model is None or not hasattr(model, "predict"):
            continue
        pred = np.asarray(model.predict(X))
        depth = int(model.get_depth()) if hasattr(model, "get_depth") else 99
        leaves = int(model.get_n_leaves()) if hasattr(model, "get_n_leaves") else 999
        simplicity = 1.0 / (1.0 + np.log1p(max(1, leaves) + max(1, depth)))
        per_class = recall_score(y, pred, labels=np.unique(y), average=None, zero_division=0)
        row = {
            "candidate": str(name),
            "fidelity": float(accuracy_score(oracle_y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "minority_recall": float(np.min(per_class)) if len(per_class) else 0.0,
            "simplicity": float(simplicity),
            "depth": depth,
            "leaves": leaves,
        }
        row["utility"] = float(
            config.fidelity_weight * row["fidelity"]
            + config.balanced_accuracy_weight * row["balanced_accuracy"]
            + config.macro_f1_weight * row["macro_f1"]
            + config.minority_recall_weight * row["minority_recall"]
            + config.simplicity_weight * row["simplicity"]
        )
        rows.append(row)
        valid.append((name, model))
    if not rows:
        raise ValueError("Nenhum candidato de árvore válido foi fornecido.")
    selected_index = max(range(len(rows)), key=lambda i: (rows[i]["utility"], -rows[i]["leaves"]))
    for idx, row in enumerate(rows):
        row["selected"] = idx == selected_index
    return valid[selected_index][1], {
        "selection_scope": "internal_training_validation",
        "test_used": False,
        "selected": rows[selected_index],
        "candidates": rows,
        "tree_refit": False,
    }


def redistill_to_crisp_tree(*args, **kwargs):
    raise RuntimeError(
        "Redestilação para outra família de árvore foi removida da produção. "
        "Use o modelo TREPAN histórico seleccionado sem conversão."
    )


__all__ = ["SoftDecisionTreeClassifier", "GlobalTreeConfig", "select_global_surrogate", "redistill_to_crisp_tree"]
