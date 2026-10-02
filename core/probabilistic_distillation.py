"""Destilação probabilística de um oráculo para árvores sem soft targets nativos."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class DistillationConfig:
    temperature: float = 2.0
    soft_weight: float = 0.35
    min_probability: float = 0.03
    hard_weight: float = 1.0
    entropy_emphasis: float = 0.5
    max_expansion_factor: float = 5.0
    true_label_weight: float = 0.45
    teacher_weight: float = 0.45
    semantic_weight: float = 0.10

    def validate(self) -> None:
        if self.temperature <= 0:
            raise ValueError("temperature deve ser > 0.")
        if not 0 <= self.soft_weight <= 1:
            raise ValueError("soft_weight deve estar em [0, 1].")
        if not 0 <= self.min_probability < 1:
            raise ValueError("min_probability deve estar em [0, 1).")
        if self.max_expansion_factor < 1:
            raise ValueError("max_expansion_factor deve ser >= 1.")
        hybrid_total = self.true_label_weight + self.teacher_weight + self.semantic_weight
        if hybrid_total <= 0:
            raise ValueError("Os pesos da destilação híbrida devem somar massa positiva.")


def temperature_scale(probabilities, temperature: float) -> np.ndarray:
    probs = np.asarray(probabilities, dtype=float)
    if probs.ndim != 2 or probs.shape[1] < 2:
        raise ValueError("predict_proba deve devolver [n_amostras, n_classes>=2].")
    if not np.isfinite(probs).all() or np.any(probs < 0):
        raise ValueError("Probabilidades do oráculo inválidas.")
    row_sum = probs.sum(axis=1, keepdims=True)
    if np.any(row_sum <= 0):
        raise ValueError("Cada linha de probabilidades deve ter massa positiva.")
    probs = np.clip(probs / row_sum, 1e-12, 1.0)
    logits = np.log(probs) / float(temperature)
    logits -= logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def expand_soft_targets(
    X, hard_labels, probabilities, classes: Sequence,
    sample_weights: Optional[Sequence] = None,
    config: DistillationConfig = DistillationConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Converte distribuições soft em linhas ponderadas, preservando hard labels."""
    config.validate()
    X_arr = np.asarray(X, dtype=float)
    y_hard = np.asarray(hard_labels)
    classes_arr = np.asarray(classes)
    if X_arr.ndim != 2 or len(X_arr) != len(y_hard):
        raise ValueError("X/hard_labels incompatíveis na destilação.")
    soft = temperature_scale(probabilities, config.temperature)
    if soft.shape != (len(X_arr), len(classes_arr)):
        raise ValueError("classes não corresponde às colunas de predict_proba.")
    base = (
        np.asarray(sample_weights, dtype=float)
        if sample_weights is not None else np.ones(len(X_arr), dtype=float)
    )
    if base.shape != (len(X_arr),) or np.any(base < 0):
        raise ValueError("sample_weights inválidos.")

    entropy = -(soft * np.log(np.clip(soft, 1e-12, 1))).sum(axis=1)
    entropy /= np.log(soft.shape[1])
    uncertainty_weight = 1.0 + float(config.entropy_emphasis) * entropy
    hard_mass = max(0.0, 1.0 - config.soft_weight) * config.hard_weight
    rows, labels, weights = [X_arr], [y_hard], [base * hard_mass]
    remaining = max(0, int(np.floor(len(X_arr) * (config.max_expansion_factor - 1))))
    candidates = []
    for class_index, class_label in enumerate(classes_arr):
        idx = np.flatnonzero(soft[:, class_index] >= config.min_probability)
        for row_index in idx:
            candidates.append((
                float(soft[row_index, class_index] * uncertainty_weight[row_index]),
                int(row_index), class_label, class_index,
            ))
    candidates.sort(key=lambda item: (-item[0], item[1], item[3]))
    selected = candidates[:remaining] if remaining else []
    if selected:
        selected_rows = np.asarray([item[1] for item in selected], dtype=int)
        selected_labels = np.asarray([item[2] for item in selected], dtype=classes_arr.dtype)
        selected_probs = np.asarray([soft[r, c] for _, r, _, c in selected])
        rows.append(X_arr[selected_rows])
        labels.append(selected_labels)
        weights.append(
            base[selected_rows] * selected_probs * config.soft_weight
            * uncertainty_weight[selected_rows]
        )
    X_out = np.vstack(rows)
    y_out = np.concatenate(labels)
    w_out = np.concatenate(weights)
    audit = {
        "method": "temperature_scaled_probability_mass",
        "temperature": float(config.temperature),
        "soft_weight": float(config.soft_weight),
        "entropy_emphasis": float(config.entropy_emphasis),
        "hard_rows": int(len(X_arr)),
        "soft_rows": int(len(selected)),
        "total_weighted_rows": int(len(X_out)),
        "mean_oracle_entropy": float(entropy.mean()),
    }
    return X_out, y_out, w_out, audit


def expand_hybrid_targets(
    X,
    true_labels,
    teacher_probabilities,
    classes: Sequence,
    *,
    semantic_confidence: Optional[Sequence] = None,
    sample_weights: Optional[Sequence] = None,
    config: DistillationConfig = DistillationConfig(),
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Combina rótulo real, distribuição do professor e confiança semântica.

    A representação por linhas ponderadas permite treinar uma árvore crisp sem
    fingir que ``DecisionTreeClassifier`` aceita distribuições como alvo.
    """
    config.validate()
    X_arr = np.asarray(X, dtype=float)
    y_true = np.asarray(true_labels)
    classes_arr = np.asarray(classes)
    if X_arr.ndim != 2 or len(X_arr) != len(y_true):
        raise ValueError("X/true_labels incompatíveis na destilação híbrida.")
    probs = temperature_scale(teacher_probabilities, config.temperature)
    if probs.shape != (len(X_arr), len(classes_arr)):
        raise ValueError("teacher_probabilities/classes incompatíveis.")
    base = np.ones(len(X_arr), dtype=float) if sample_weights is None else np.asarray(
        sample_weights, dtype=float,
    )
    semantic = np.ones(len(X_arr), dtype=float) if semantic_confidence is None else np.asarray(
        semantic_confidence, dtype=float,
    )
    if base.shape != (len(X_arr),) or semantic.shape != (len(X_arr),):
        raise ValueError("Pesos base/semânticos incompatíveis.")
    semantic = np.clip(semantic, 0.0, 1.0)
    total = config.true_label_weight + config.teacher_weight + config.semantic_weight
    w_true = base * (config.true_label_weight / total)
    # A massa semântica reforça apenas exemplos com conceitos consistentes.
    w_true += base * semantic * (config.semantic_weight / total)
    rows = [X_arr]
    labels = [y_true]
    weights = [w_true]
    teacher_mass = config.teacher_weight / total
    max_soft = max(0, int(len(X_arr) * (config.max_expansion_factor - 1)))
    candidates = []
    entropy = -(probs * np.log(np.clip(probs, 1e-12, 1))).sum(axis=1)
    entropy /= np.log(probs.shape[1])
    for class_idx, class_label in enumerate(classes_arr):
        valid = np.flatnonzero(probs[:, class_idx] >= config.min_probability)
        for row_idx in valid:
            mass = probs[row_idx, class_idx] * (1.0 + config.entropy_emphasis * entropy[row_idx])
            candidates.append((float(mass), int(row_idx), class_idx, class_label))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))
    selected = candidates[:max_soft]
    if selected:
        idx = np.asarray([item[1] for item in selected], dtype=int)
        cls_idx = np.asarray([item[2] for item in selected], dtype=int)
        rows.append(X_arr[idx])
        labels.append(np.asarray([item[3] for item in selected], dtype=classes_arr.dtype))
        weights.append(
            base[idx] * probs[idx, cls_idx] * teacher_mass
            * (1.0 + config.entropy_emphasis * entropy[idx])
        )
    return np.vstack(rows), np.concatenate(labels), np.concatenate(weights), {
        "method": "hybrid_true_soft_semantic_distillation",
        "temperature": float(config.temperature),
        "true_label_weight": float(config.true_label_weight),
        "teacher_weight": float(config.teacher_weight),
        "semantic_weight": float(config.semantic_weight),
        "hard_rows": int(len(X_arr)),
        "soft_rows": int(len(selected)),
        "test_used": False,
    }


def constrain_synthetic_samples(
    samples,
    *,
    min_samples: int,
    ontology_graph=None,
    feature_names: Optional[Sequence[str]] = None,
) -> Tuple[np.ndarray, dict]:
    """Filtra amostras sintéticas que violam restrições declaradas na OWL.

    ``samples`` são os vectores estatísticos (KDE/frequências) já gerados na
    fronteira de decisão. As linhas válidas segundo
    ``ontology_graph.sample_validity_mask`` vêm primeiro, pela ordem original.
    Fallback transparente: se houver menos de ``min_samples`` linhas válidas, o
    lote é completado com as amostras estatísticas padrão rejeitadas (também
    pela ordem original), para que o crescimento da árvore nunca pare.
    """
    X = np.asarray(samples, dtype=float)
    needed = max(0, int(min_samples))
    audit = {
        "method": "ontology_constrained_synthetic_sampling",
        "candidates": int(len(X)),
        "min_samples": needed,
        "ontology_active": False,
        "valid": int(len(X)),
        "rejected": 0,
        "fallback_filled": 0,
    }
    active = bool(ontology_graph is not None and getattr(ontology_graph, "is_active", False))
    if not active or X.ndim != 2 or len(X) == 0 or feature_names is None:
        return X, audit
    audit["ontology_active"] = True
    valid = np.asarray(ontology_graph.sample_validity_mask(X, feature_names), dtype=bool)
    if valid.shape != (len(X),):
        raise ValueError("sample_validity_mask devolveu forma incompatível com as amostras.")
    kept = X[valid]
    rejected = X[~valid]
    fill = max(0, needed - len(kept))
    if fill:
        kept = np.vstack([kept, rejected[:fill]]) if len(kept) else rejected[:fill]
    audit.update({
        "valid": int(valid.sum()),
        "rejected": int((~valid).sum()),
        "fallback_filled": int(min(fill, len(rejected))),
    })
    return kept, audit


def build_semantic_query_projector(
    ontology_graph,
    feature_names: Sequence[str],
    reference_X=None,
    *,
    semantic_query_projection: bool = True,
    base_projector=None,
):
    """Projector de membership queries para o espaço conceptual da OWL.

    Cada query gerada na fronteira do nó é primeiro projectada para o domínio
    OWL (arredondamento xsd:integer + recorte aos limites declarados) e só
    depois passa pelo ``base_projector`` (ex.: recomposição de colunas onto_*)
    e pelo oráculo. Devolve ``(projector, constraints, audit)``; o projector é
    ``base_projector`` inalterado quando não há restrições OWL aplicáveis.
    """
    audit = {
        "semantic_query_projection": bool(semantic_query_projection),
        "enabled": False,
        "reason": "disabled",
    }
    if not semantic_query_projection:
        return base_projector, None, audit
    if ontology_graph is None or not getattr(ontology_graph, "is_active", False):
        audit["reason"] = "ontology_inactive"
        return base_projector, None, audit
    constraints = ontology_graph.domain_constraints(feature_names, reference_X)
    audit.update(constraints.summary())
    if not constraints.is_active:
        audit["reason"] = "no_applicable_owl_domain_constraints"
        return base_projector, None, audit

    def projector(rows):
        projected = constraints.project(rows)
        return projected if base_projector is None else base_projector(projected)

    audit.update({"enabled": True, "reason": "owl_domain_projection"})
    return projector, constraints, audit


__all__ = [
    "DistillationConfig", "temperature_scale", "expand_soft_targets",
    "expand_hybrid_targets", "constrain_synthetic_samples",
    "build_semantic_query_projector",
]
