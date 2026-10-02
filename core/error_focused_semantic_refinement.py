"""Utilitários do Error-Focused Semantic Refinement (EFSR) V9.2.

Este módulo é deliberadamente agnóstico ao dataset. Ele não contém nomes de
atributos, datasets, classes ou thresholds de domínio. A informação de classe
usada aqui é exclusivamente o rótulo do oráculo já conhecido no conjunto de
TREINO/queries do TREPAN; nunca recebe o conjunto de teste externo.

O EFSR mede onde a folha corrente do surrogate discorda do MLP, estima quais
features distinguem melhor os erros locais e permite auditar se um candidato
semântico melhora de facto a fidelidade local antes de substituir o split
puramente estatístico.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ErrorRegionProfile:
    node_id: int
    reach: float
    sample_count: int
    disagreement_count: int
    disagreement_rate: float
    fidelity: float
    mean_uncertainty: float
    disagreement_uncertainty: float
    semantic_opportunity: float
    feature_error_scores: tuple[float, ...]
    semantic_feature_scores: tuple[float, ...]
    anchor_indices: tuple[int, ...]
    probability_source: str = "unavailable"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalized_entropy(probabilities: np.ndarray) -> np.ndarray:
    """Entropia normalizada em [0,1] por linha.

    0 = oráculo muito confiante; 1 = distribuição uniforme. Valores inválidos
    são normalizados de forma conservadora em vez de propagarem NaN.
    """
    p = np.asarray(probabilities, dtype=float)
    if p.ndim != 2 or p.shape[1] < 2:
        return np.zeros(len(p), dtype=float)
    p = np.clip(p, 1e-12, None)
    row_sum = p.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    p = p / row_sum
    ent = -np.sum(p * np.log(p), axis=1)
    denom = float(np.log(p.shape[1]))
    if denom <= 0:
        return np.zeros(len(p), dtype=float)
    return np.clip(ent / denom, 0.0, 1.0)


def robust_error_feature_scores(X: np.ndarray, disagreement_mask: np.ndarray) -> np.ndarray:
    """Pontuação agnóstica de associação feature↔erro local.

    Usa diferença robusta de medianas entre casos em que TREPAN e MLP discordam
    e casos em que concordam, normalizada por IQR/MAD-like scale. O resultado é
    reescalado para [0,1] dentro da região e serve apenas para priorização.
    """
    X = np.asarray(X, dtype=float)
    mask = np.asarray(disagreement_mask, dtype=bool).reshape(-1)
    if X.ndim != 2 or len(mask) != len(X) or X.shape[1] == 0:
        return np.zeros(X.shape[1] if X.ndim == 2 else 0, dtype=float)
    if mask.sum() < 2 or (~mask).sum() < 2:
        return np.zeros(X.shape[1], dtype=float)

    err = X[mask]
    ok = X[~mask]
    med_err = np.nanmedian(err, axis=0)
    med_ok = np.nanmedian(ok, axis=0)
    q75 = np.nanpercentile(X, 75, axis=0)
    q25 = np.nanpercentile(X, 25, axis=0)
    iqr = np.asarray(q75 - q25, dtype=float)
    std = np.nanstd(X, axis=0)
    scale = np.where(iqr > 1e-9, iqr, np.where(std > 1e-9, std, 1.0))
    raw = np.abs(med_err - med_ok) / scale
    raw = np.where(np.isfinite(raw), raw, 0.0)
    if not np.any(raw > 0):
        return np.zeros_like(raw)
    # Compressão robusta e normalização; tanh reduz dominância de outliers.
    raw = np.tanh(raw)
    top = float(np.max(raw))
    return np.clip(raw / max(top, 1e-12), 0.0, 1.0)


def semantic_support_vector(
    n_features: int,
    *,
    feature_weights: Optional[Sequence[float]] = None,
    feature_groups: Optional[Sequence[Optional[str]]] = None,
    relatedness_matrix: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Força estrutural por feature, sem qualquer conhecimento do target."""
    n = int(n_features)
    if n <= 0:
        return np.zeros(0, dtype=float)
    support = np.zeros(n, dtype=float)

    if feature_weights is not None:
        w = np.asarray(feature_weights, dtype=float).reshape(-1)
        if len(w) == n and np.isfinite(w).all():
            centred = np.abs(w - 1.0)
            if np.any(centred > 0):
                centred = centred / max(float(np.max(centred)), 1e-12)
                support = np.maximum(support, centred)

    if feature_groups is not None and len(feature_groups) == n:
        counts: dict[str, int] = {}
        for g in feature_groups:
            if g not in (None, "") and str(g).lower() not in {"general", "none"}:
                counts[str(g)] = counts.get(str(g), 0) + 1
        for i, g in enumerate(feature_groups):
            if g not in (None, "") and counts.get(str(g), 0) > 1:
                support[i] = max(support[i], min(1.0, (counts[str(g)] - 1) / 3.0 + 0.34))

    if relatedness_matrix is not None:
        rel = np.asarray(relatedness_matrix, dtype=float)
        if rel.shape == (n, n) and np.isfinite(rel).all():
            off = rel.copy()
            np.fill_diagonal(off, 0.0)
            connectivity = np.mean(np.clip(off, 0.0, 1.0), axis=1)
            mx = float(np.max(connectivity)) if len(connectivity) else 0.0
            if mx > 0:
                connectivity = connectivity / mx
                support = np.maximum(support, connectivity)

    return np.clip(support, 0.0, 1.0)


def combine_error_and_semantics(
    error_scores: np.ndarray,
    semantic_support: np.ndarray,
    *,
    semantic_weight: float = 0.50,
) -> np.ndarray:
    err = np.asarray(error_scores, dtype=float)
    sem = np.asarray(semantic_support, dtype=float)
    if err.shape != sem.shape:
        raise ValueError("error_scores e semantic_support devem ter a mesma forma.")
    alpha = float(np.clip(semantic_weight, 0.0, 1.0))
    combined = err * ((1.0 - alpha) + alpha * (1.0 + sem))
    if not np.any(combined > 0):
        return np.zeros_like(combined)
    return np.clip(combined / max(float(np.max(combined)), 1e-12), 0.0, 1.0)


def split_surrogate_fidelity(y: np.ndarray, mask: np.ndarray, classes: Sequence[Any]) -> float:
    """Fidelidade local de uma divisão quando cada ramo prevê a classe majoritária."""
    y = np.asarray(y)
    mask = np.asarray(mask, dtype=bool)
    classes = np.asarray(list(classes))
    if len(y) == 0 or len(mask) != len(y):
        return 0.0
    pred = np.empty(len(y), dtype=object)
    for branch in (False, True):
        idx = mask if branch else ~mask
        if not np.any(idx):
            return 0.0
        counts = np.asarray([(y[idx] == c).sum() for c in classes], dtype=int)
        pred[idx] = classes[int(np.argmax(counts))]
    return float(np.mean(pred == y))


def nearest_anchor_similarity(
    candidates: np.ndarray,
    anchors: np.ndarray,
    *,
    feature_indices: Sequence[int],
    scales: np.ndarray,
) -> np.ndarray:
    """Similaridade [0,1] a regiões de erro conhecidas, sem consultar o oráculo."""
    C = np.asarray(candidates, dtype=float)
    A = np.asarray(anchors, dtype=float)
    if len(C) == 0 or len(A) == 0:
        return np.zeros(len(C), dtype=float)
    idx = np.asarray(list(feature_indices), dtype=int)
    if idx.size == 0:
        idx = np.arange(C.shape[1], dtype=int)
    scl = np.asarray(scales, dtype=float)[idx]
    scl = np.where(scl > 1e-9, scl, 1.0)
    # Evita tensor gigantesco: processa anchors um a um e guarda o melhor match.
    best = np.zeros(len(C), dtype=float)
    for anchor in A:
        dist = np.mean(np.abs((C[:, idx] - anchor[idx]) / scl), axis=1)
        best = np.maximum(best, np.exp(-dist))
    return np.clip(best, 0.0, 1.0)


__all__ = [
    "ErrorRegionProfile",
    "normalized_entropy",
    "robust_error_feature_scores",
    "semantic_support_vector",
    "combine_error_and_semantics",
    "split_surrogate_fidelity",
    "nearest_anchor_similarity",
]
