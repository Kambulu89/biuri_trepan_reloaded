"""Contrafactuais de fronteira plausíveis gerados apenas com dados de treino."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class PlausibleCFConfig:
    max_counterfactuals: int = 128
    binary_search_steps: int = 12
    max_scaled_distance: float = 6.0
    random_state: int = 42


def generate_plausible_boundary_counterfactuals(
    oracle,
    X_train,
    *,
    config: PlausibleCFConfig = PlausibleCFConfig(),
    logical_validator: Optional[Callable[[np.ndarray], bool]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Interpola vizinhos de classes do oráculo diferentes e localiza a fronteira.

    Não usa linhas do teste. A plausibilidade combina proximidade ao manifold,
    distância factual e validação lógica opcional.
    """
    X = np.asarray(X_train, dtype=float)
    if X.ndim != 2 or len(X) < 4:
        raise ValueError("Contrafactuais plausíveis exigem matriz de treino 2D.")
    if int(config.max_counterfactuals) <= 0:
        return np.empty((0, X.shape[1])), np.empty(0), {
            "scope": "training_only",
            "test_used": False,
            "generated": 0,
            "reason": "disabled_for_matched_control",
        }
    labels = np.asarray(oracle.predict(X))
    if len(np.unique(labels)) < 2:
        return np.empty((0, X.shape[1])), np.empty(0), {
            "scope": "training_only", "generated": 0, "reason": "single_oracle_class",
        }
    scale = np.std(X, axis=0) + 1e-9
    normalized = X / scale
    rng = np.random.default_rng(config.random_state)
    order = rng.permutation(len(X))
    candidates, weights, distances = [], [], []
    reference = normalized
    if len(reference) > 600:
        reference = reference[rng.choice(len(reference), 600, replace=False)]
    for anchor_idx in order:
        opposite = np.flatnonzero(labels != labels[anchor_idx])
        if not len(opposite):
            continue
        distance = np.linalg.norm(normalized[opposite] - normalized[anchor_idx], axis=1)
        neighbor_idx = int(opposite[int(np.argmin(distance))])
        if float(distance.min()) > config.max_scaled_distance:
            continue
        low, high = X[anchor_idx].copy(), X[neighbor_idx].copy()
        source_label = labels[anchor_idx]
        for _ in range(config.binary_search_steps):
            middle = (low + high) / 2.0
            if oracle.predict(middle.reshape(1, -1))[0] == source_label:
                low = middle
            else:
                high = middle
        cf = high
        if oracle.predict(cf.reshape(1, -1))[0] == source_label:
            continue
        if logical_validator is not None and not bool(logical_validator(cf)):
            continue
        scaled_cf = cf / scale
        manifold_distance = float(np.min(np.linalg.norm(reference - scaled_cf, axis=1)))
        factual_distance = float(np.linalg.norm((cf - X[anchor_idx]) / scale))
        plausibility = float(np.exp(-manifold_distance) * np.exp(-0.15 * factual_distance))
        if plausibility <= 1e-6:
            continue
        candidates.append(cf)
        weights.append(plausibility)
        distances.append(factual_distance)
        if len(candidates) >= config.max_counterfactuals:
            break
    if not candidates:
        return np.empty((0, X.shape[1])), np.empty(0), {
            "scope": "training_only", "generated": 0,
        }
    array = np.asarray(candidates, dtype=float)
    weight_array = np.asarray(weights, dtype=float)
    # Deduplica sem inventar perturbações adicionais.
    _, unique_idx = np.unique(np.round(array, 10), axis=0, return_index=True)
    unique_idx = np.sort(unique_idx)
    array, weight_array = array[unique_idx], weight_array[unique_idx]
    return array, weight_array, {
        "scope": "training_only", "test_used": False,
        "method": "opposite_oracle_neighbor_boundary_search",
        "generated": int(len(array)),
        "mean_plausibility": float(weight_array.mean()),
        "mean_scaled_distance": float(np.mean(np.asarray(distances)[unique_idx])),
        "logical_validator_used": logical_validator is not None,
    }


__all__ = ["PlausibleCFConfig", "generate_plausible_boundary_counterfactuals"]
