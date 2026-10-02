"""Selecção multiobjectivo de um TREPAN já treinado, sem refit de outra família."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score


@dataclass(frozen=True)
class MultiObjectiveConfig:
    fidelity_tolerance: float = 0.01
    accuracy_tolerance: float = 0.01
    balanced_accuracy_tolerance: float = 0.01
    macro_f1_tolerance: float = 0.01
    robustness_epsilon: float = 0.01
    max_pruning_candidates: int = 1
    random_state: int = 42


def _dominates(a: Dict[str, float], b: Dict[str, float]) -> bool:
    keys = ("balanced_accuracy", "macro_f1", "minority_recall", "fidelity", "robustness", "accuracy", "simplicity")
    return all(a[k] >= b[k] - 1e-12 for k in keys) and any(a[k] > b[k] + 1e-12 for k in keys)


def pareto_front(rows: Sequence[Dict[str, float]]) -> list[int]:
    return [i for i, row in enumerate(rows) if not any(j != i and _dominates(other, row) for j, other in enumerate(rows))]


def select_multiobjective_pruned_tree(
    initial_tree,
    oracle,
    X_fit,
    y_fit,
    sample_weights,
    X_reference,
    *,
    semantic_feature_indices: Optional[Iterable[int]] = None,
    y_reference_true=None,
    config: MultiObjectiveConfig = MultiObjectiveConfig(),
):
    """Avalia o modelo histórico sem gerar candidatos de poda de outra família."""
    if initial_tree is None or not hasattr(initial_tree, "predict"):
        raise ValueError("É necessário fornecer um TREPAN treinado.")
    X = np.asarray(X_reference, dtype=float)
    oracle_y = np.asarray(oracle.predict(X))
    pred = np.asarray(initial_tree.predict(X))
    true = None if y_reference_true is None else np.asarray(y_reference_true)
    rng = np.random.default_rng(config.random_state)
    Xfit = np.asarray(X_fit, dtype=float)
    scale = np.std(Xfit, axis=0) + 1e-9
    perturbed = X + rng.normal(0, config.robustness_epsilon, X.shape) * scale
    pred_perturbed = np.asarray(initial_tree.predict(perturbed))
    nodes = int(getattr(initial_tree, "node_count_", 0) or 0)
    if not nodes and hasattr(initial_tree, "tree_"):
        nodes = int(getattr(initial_tree.tree_, "node_count", 1))
    nodes = max(1, nodes)
    depth = int(initial_tree.get_depth()) if hasattr(initial_tree, "get_depth") else 0
    leaves = int(initial_tree.get_n_leaves()) if hasattr(initial_tree, "get_n_leaves") else 1
    if true is None:
        acc = ba = macro = minority = 0.0
    else:
        per_class = recall_score(true, pred, labels=np.unique(true), average=None, zero_division=0)
        acc = float(accuracy_score(true, pred))
        ba = float(balanced_accuracy_score(true, pred))
        macro = float(f1_score(true, pred, average="macro", zero_division=0))
        minority = float(np.min(per_class)) if len(per_class) else 0.0
    row = {
        "candidate": "historical_trepan",
        "fidelity": float(accuracy_score(oracle_y, pred)),
        "robustness": float(accuracy_score(pred, pred_perturbed)),
        "balanced_accuracy": ba,
        "macro_f1": macro,
        "minority_recall": minority,
        "accuracy": acc,
        "simplicity": float(1.0 / (1.0 + np.log1p(nodes))),
        "nodes": nodes,
        "depth": depth,
        "leaves": leaves,
        "pareto": True,
        "selected": True,
    }
    return initial_tree, {
        "selection_scope": "internal_training_validation",
        "test_used": False,
        "pruning_family": "historical_trepan_only",
        "pareto_candidates": [row],
        "selected": row,
    }


__all__ = ["MultiObjectiveConfig", "pareto_front", "select_multiobjective_pruned_tree"]
