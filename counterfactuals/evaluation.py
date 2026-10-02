"""Avaliação formal e comparativa de contrafactuais já gerados.

Este módulo não contém geradores. Ele enriquece o resultado do motor único com
as métricas do módulo experimental antigo, evitando implementações paralelas de
DiCE/CLEAR/CoGS/LORE.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np


EPS = 1e-12


def _predict(model: Any, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.asarray(model.predict(array)).reshape(-1)


def _probability_margin(model: Any, vector: np.ndarray, target: Any) -> Optional[float]:
    if not hasattr(model, "predict_proba"):
        return None
    try:
        probabilities = np.asarray(model.predict_proba(vector.reshape(1, -1)), dtype=float)[0]
    except Exception:
        return None
    classes = getattr(model, "classes_", None)
    if classes is None:
        classes = getattr(getattr(model, "model", None), "classes_", None)
    if classes is None:
        target_index = int(target) if str(target).lstrip("-").isdigit() else int(np.argmax(probabilities))
    else:
        matches = np.where(np.asarray(classes) == target)[0]
        if not len(matches):
            return None
        target_index = int(matches[0])
    if target_index < 0 or target_index >= len(probabilities):
        return None
    alternatives = np.delete(probabilities, target_index)
    strongest_alternative = float(np.max(alternatives)) if len(alternatives) else 0.0
    return float(probabilities[target_index] - strongest_alternative)


def _mahalanobis(vector: np.ndarray, reference: np.ndarray) -> Optional[float]:
    if len(reference) < 2:
        return None
    mean = np.mean(reference, axis=0)
    covariance = np.atleast_2d(np.cov(reference, rowvar=False))
    if covariance.shape != (reference.shape[1], reference.shape[1]):
        return None
    inverse = np.linalg.pinv(covariance + np.eye(covariance.shape[0]) * 1e-8)
    difference = vector - mean
    return float(np.sqrt(max(float(difference @ inverse @ difference), 0.0)))


def _nearest_target_distance(
    vector: np.ndarray, X: np.ndarray, y: np.ndarray, target: Any, scale: np.ndarray,
) -> Optional[float]:
    points = X[y == target]
    if not len(points):
        return None
    distances = np.linalg.norm((points - vector) / scale, axis=1)
    return float(np.min(distances))


def _distance_to_boundary(
    model: Any,
    vector: np.ndarray,
    target: Any,
    scale: np.ndarray,
    *,
    seed: int,
    directions: int = 24,
    max_radius: float = 3.0,
) -> Optional[float]:
    """Procura a fronteira em direcções normalizadas e refina por bissecção."""
    if _predict(model, vector)[0] != target:
        return 0.0
    rng = np.random.RandomState(seed)
    best: Optional[float] = None
    for _ in range(max(4, int(directions))):
        direction = rng.normal(size=len(vector))
        norm = float(np.linalg.norm(direction))
        if norm < EPS:
            continue
        direction /= norm
        low, high = 0.0, 0.05
        changed = False
        while high <= max_radius:
            probe = vector + direction * high * scale
            if _predict(model, probe)[0] != target:
                changed = True
                break
            low, high = high, high * 2.0
        if not changed:
            continue
        for _ in range(16):
            middle = (low + high) / 2.0
            probe = vector + direction * middle * scale
            if _predict(model, probe)[0] == target:
                low = middle
            else:
                high = middle
        best = high if best is None else min(best, high)
    return best


def evaluate_generation_result(
    result: Mapping[str, Any],
    oracle: Any,
    X_reference: Any,
    y_reference: Any,
    *,
    seed: int = 42,
) -> Dict[str, Any]:
    """Calcula métricas formais por candidato, resumo e comparações por método."""
    original = np.asarray(result.get("original_instance"), dtype=float).reshape(-1)
    X = np.asarray(X_reference, dtype=float)
    y = np.asarray(y_reference).reshape(-1)
    if X.ndim != 2 or X.shape[1] != len(original) or len(X) != len(y):
        raise ValueError("O resultado e o dataset de referência estão desalinhados.")
    scale = np.std(X, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    factual = _predict(oracle, original)[0]
    rows: List[Dict[str, Any]] = []
    for index, candidate in enumerate(result.get("candidates") or []):
        vector = np.asarray(candidate.get("vector"), dtype=float).reshape(-1)
        if len(vector) != len(original):
            continue
        prediction = _predict(oracle, vector)[0]
        delta = vector - original
        changed = np.abs(delta) > 1e-6
        validity = bool(prediction != factual and candidate.get("metrics", {}).get("validity", False))
        row = {
            "candidate_index": index + 1,
            "method": candidate.get("method", "desconhecido"),
            "validity": validity,
            "proximity_l1": float(np.sum(np.abs(delta))),
            "proximity_l2": float(np.linalg.norm(delta)),
            "proximity_weighted_l1": float(np.sum(np.abs(delta) / scale)),
            "sparsity_l0": int(np.sum(changed)),
            "sparsity_normalized": float(1.0 - np.mean(changed)) if len(changed) else 0.0,
            "plausibility_nn_target": _nearest_target_distance(vector, X, y, prediction, scale),
            "plausibility_mahalanobis": _mahalanobis(vector, X[y == prediction]),
            "probability_margin": _probability_margin(oracle, vector, prediction),
            "distance_to_boundary": _distance_to_boundary(
                oracle, vector, prediction, scale, seed=seed + index,
            ),
        }
        rows.append(row)

    numeric_keys = (
        "validity", "proximity_l1", "proximity_l2", "proximity_weighted_l1",
        "sparsity_l0", "sparsity_normalized", "plausibility_nn_target",
        "plausibility_mahalanobis", "probability_margin", "distance_to_boundary",
    )
    summary: Dict[str, Any] = {"n_candidates": len(rows)}
    for key in numeric_keys:
        values = [float(row[key]) for row in rows if row.get(key) is not None and np.isfinite(row[key])]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    vectors = [np.asarray(candidate.get("vector"), dtype=float) for candidate in result.get("candidates") or []]
    pairwise = [float(np.linalg.norm((a - b) / scale)) for a, b in combinations(vectors, 2)]
    summary["diversity_weighted_l2"] = float(np.mean(pairwise)) if pairwise else 0.0

    comparisons = compare_methods(rows, metric="proximity_weighted_l1")
    return {
        "validity_definition": "predição do oráculo muda e alcança a classe desejada",
        "robustness_margin_definition": (
            "probabilidade da classe prevista menos a maior probabilidade alternativa; válida em multiclasse"
        ),
        "per_candidate": rows,
        "summary": summary,
        "method_comparisons": comparisons,
    }


def compare_methods(rows: Sequence[Mapping[str, Any]], metric: str) -> List[Dict[str, Any]]:
    """Wilcoxon pareado e Cohen d quando dois métodos têm observações suficientes."""
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get(metric)
        if value is not None and np.isfinite(value):
            grouped.setdefault(str(row.get("method", "desconhecido")), []).append(float(value))
    output: List[Dict[str, Any]] = []
    for first, second in combinations(sorted(grouped), 2):
        a, b = np.asarray(grouped[first]), np.asarray(grouped[second])
        n = min(len(a), len(b))
        if n < 2:
            continue
        statistic = p_value = None
        try:
            from scipy.stats import wilcoxon
            test = wilcoxon(a[:n], b[:n])
            statistic, p_value = float(test.statistic), float(test.pvalue)
        except Exception:
            pass
        pooled_denominator = len(a) + len(b) - 2
        if pooled_denominator > 0:
            pooled = np.sqrt(
                ((len(a) - 1) * np.var(a, ddof=1) + (len(b) - 1) * np.var(b, ddof=1))
                / pooled_denominator
            )
            effect = float((np.mean(a) - np.mean(b)) / pooled) if pooled > EPS else 0.0
        else:
            effect = None
        output.append({
            "metric": metric, "method_a": first, "method_b": second,
            "n_pairs": n, "wilcoxon_statistic": statistic,
            "p_value": p_value, "cohens_d": effect,
        })
    return output


__all__ = ["evaluate_generation_result", "compare_methods"]
