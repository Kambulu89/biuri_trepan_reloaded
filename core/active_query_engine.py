"""Consultas ativas por discordância, incerteza e fronteiras contrafactuais."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, recall_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class ActiveQueryConfig:
    iterations: int = 3
    budget_per_iteration: int = 128
    pool_multiplier: int = 8
    disagreement_weight: float = 0.50
    uncertainty_weight: float = 0.30
    counterfactual_weight: float = 0.20
    diversity_weight: float = 0.15
    c45_disagreement_weight: float = 0.25
    plausibility_weight: float = 0.15
    real_label_error_weight: float = 0.30
    minority_weight: float = 0.20
    semantic_gap_weight: float = 0.20
    perturbation_scale: float = 0.15
    random_state: int = 42

    def validate(self) -> None:
        if self.iterations < 0 or self.budget_per_iteration < 1:
            raise ValueError("Configuração de consultas ativas inválida.")
        if self.pool_multiplier < 1 or self.perturbation_scale <= 0:
            raise ValueError("pool_multiplier/perturbation_scale inválidos.")


def _oracle_probabilities(oracle, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(oracle.predict(X))
    if hasattr(oracle, "predict_proba"):
        probs = np.asarray(oracle.predict_proba(X), dtype=float)
        probs = np.clip(probs, 1e-12, 1.0)
        probs /= probs.sum(axis=1, keepdims=True)
    else:
        classes = np.asarray(getattr(oracle, "classes_", np.unique(labels)))
        mapping = {label: idx for idx, label in enumerate(classes)}
        probs = np.zeros((len(X), len(classes)), dtype=float)
        probs[np.arange(len(X)), [mapping[label] for label in labels]] = 1.0
    return labels, probs


def _diverse_top_k(X: np.ndarray, scores: np.ndarray, k: int, weight: float) -> np.ndarray:
    order = np.argsort(-scores, kind="stable")
    if len(order) <= k or weight <= 0:
        return order[:k]
    scale = np.std(X, axis=0) + 1e-9
    normalized = X / scale
    selected = [int(order[0])]
    remaining = set(map(int, order[1:]))
    while remaining and len(selected) < k:
        candidates = np.asarray(sorted(remaining), dtype=int)
        distances = np.min(
            np.linalg.norm(
                normalized[candidates, None, :] - normalized[np.asarray(selected)][None, :, :],
                axis=2,
            ), axis=1,
        )
        diversity = distances / (distances.max() + 1e-12)
        combined = scores[candidates] + weight * diversity
        chosen = int(candidates[int(np.argmax(combined))])
        selected.append(chosen)
        remaining.remove(chosen)
    return np.asarray(selected, dtype=int)


def refine_with_active_queries(
    initial_tree,
    oracle,
    X_fit,
    y_fit,
    sample_weights,
    X_reference,
    *,
    counterfactual_X: Optional[np.ndarray] = None,
    counterfactual_weights: Optional[Sequence] = None,
    c45_baseline=None,
    original_oracle=None,
    ontological_oracle=None,
    semantic_feature_indices: Optional[Sequence[int]] = None,
    y_reference_true: Optional[Sequence] = None,
    config: ActiveQueryConfig = ActiveQueryConfig(),
):
    """Refina uma árvore sem consultar a partição de teste externa."""
    config.validate()
    X_fit = np.asarray(X_fit, dtype=float)
    y_fit = np.asarray(y_fit)
    weights = np.asarray(sample_weights, dtype=float)
    X_ref = np.asarray(X_reference, dtype=float)
    if X_fit.ndim != 2 or X_ref.ndim != 2 or X_fit.shape[1] != X_ref.shape[1]:
        raise ValueError("Espaços incompatíveis no motor de consultas ativas.")
    if len(y_fit) != len(X_fit) or weights.shape != (len(X_fit),):
        raise ValueError("y_fit/sample_weights incompatíveis.")
    oracle_ref, _ = _oracle_probabilities(oracle, X_ref)
    y_ref_true = None if y_reference_true is None else np.asarray(y_reference_true)
    if y_ref_true is not None and len(y_ref_true) != len(X_ref):
        raise ValueError("y_reference_true incompatível com X_reference.")
    ref_indices = np.arange(len(X_ref))
    if len(X_ref) >= 10:
        try:
            train_idx, val_idx = train_test_split(
                ref_indices, test_size=0.2, random_state=config.random_state,
                stratify=oracle_ref,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                ref_indices, test_size=0.2, random_state=config.random_state,
            )
    else:
        train_idx = val_idx = ref_indices
    ref_train, ref_val = X_ref[train_idx], X_ref[val_idx]
    val_labels, _ = _oracle_probabilities(oracle, ref_val)
    val_true = None if y_ref_true is None else y_ref_true[val_idx]
    best = initial_tree
    best_fidelity = float(accuracy_score(val_labels, best.predict(ref_val)))
    def real_metrics(target, predictions):
        if target is None:
            return None
        recalls = recall_score(
            target, predictions, labels=np.unique(target), average=None,
            zero_division=0,
        )
        return {
            "accuracy": float(accuracy_score(target, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(target, predictions)),
            "macro_f1": float(f1_score(
                target, predictions, average="macro", zero_division=0,
            )),
            "minority_recall": float(np.min(recalls)),
        }
    best_real = real_metrics(val_true, best.predict(ref_val))
    best_accuracy = None if best_real is None else best_real["accuracy"]
    initial_fidelity = best_fidelity
    rng = np.random.default_rng(config.random_state)
    low = np.nanquantile(ref_train, 0.01, axis=0)
    high = np.nanquantile(ref_train, 0.99, axis=0)
    scale = np.nanstd(ref_train, axis=0) + 1e-9
    cf = None
    cf_weights = None
    if counterfactual_X is not None:
        cf = np.asarray(counterfactual_X, dtype=float)
        if cf.ndim != 2 or cf.shape[1] != X_fit.shape[1]:
            raise ValueError("Contrafactuais têm schema incompatível com a árvore.")
        cf_weights = np.ones(len(cf), dtype=float) if counterfactual_weights is None else np.asarray(
            counterfactual_weights, dtype=float,
        )
        if cf_weights.shape != (len(cf),) or np.any(cf_weights < 0):
            raise ValueError("Pesos de plausibilidade contrafactual inválidos.")
    trace = []
    query_X, query_y, query_w = X_fit, y_fit, weights
    for iteration in range(config.iterations):
        pool_size = max(config.budget_per_iteration, config.budget_per_iteration * config.pool_multiplier)
        anchors = ref_train[rng.integers(0, len(ref_train), size=pool_size)]
        pool = np.clip(
            anchors + rng.normal(0.0, config.perturbation_scale, size=anchors.shape) * scale,
            low, high,
        )
        cf_mask = np.zeros(len(pool), dtype=float)
        if cf is not None and len(cf):
            take = min(len(cf), max(1, pool_size // 4))
            chosen_cf = rng.choice(len(cf), size=take, replace=len(cf) < take)
            pool[:take] = cf[chosen_cf]
            cf_mask[:take] = np.clip(cf_weights[chosen_cf], 0.0, 1.0)
        oracle_labels, probs = _oracle_probabilities(oracle, pool)
        tree_labels = np.asarray(best.predict(pool))
        disagreement = (tree_labels != oracle_labels).astype(float)
        c45_disagreement = np.zeros(len(pool), dtype=float)
        if c45_baseline is not None:
            expected = getattr(c45_baseline, 'n_features_in_', pool.shape[1])
            if int(expected) > pool.shape[1]:
                raise ValueError("C4.5 espera mais features do que o pool activo.")
            c45_labels = np.asarray(c45_baseline.predict(pool[:, : int(expected)]))
            c45_disagreement = (c45_labels != oracle_labels).astype(float)
        entropy = -(probs * np.log(probs)).sum(axis=1) / np.log(probs.shape[1])
        # Plausibilidade por distância ao manifold de treino, sem rótulos de teste.
        reference = ref_train
        if len(reference) > 512:
            reference = reference[rng.choice(len(reference), size=512, replace=False)]
        norm_pool = pool / scale
        norm_ref = reference / scale
        min_distance = np.min(
            np.linalg.norm(norm_pool[:, None, :] - norm_ref[None, :, :], axis=2), axis=1,
        )
        plausibility = np.exp(-min_distance)
        real_label_error = np.zeros(len(pool), dtype=float)
        minority_score = np.zeros(len(pool), dtype=float)
        if y_ref_true is not None:
            distances = np.linalg.norm(
                norm_pool[:, None, :] - norm_ref[None, :, :], axis=2,
            )
            nearest = np.argmin(distances, axis=1)
            proxy_true = y_ref_true[train_idx][nearest]
            real_label_error = (tree_labels != proxy_true).astype(float)
            labels, counts = np.unique(y_ref_true[train_idx], return_counts=True)
            class_score = {
                label: float(counts.max() / count) for label, count in zip(labels, counts)
            }
            minority_score = np.asarray([class_score[label] for label in proxy_true])
            minority_score /= max(1.0, float(minority_score.max()))
        semantic_gap = np.zeros(len(pool), dtype=float)
        if original_oracle is not None and ontological_oracle is not None:
            original_labels, _ = _oracle_probabilities(original_oracle, pool)
            ontology_labels, _ = _oracle_probabilities(ontological_oracle, pool)
            semantic_gap = (original_labels != ontology_labels).astype(float)
        elif semantic_feature_indices:
            semantic = np.asarray(list(semantic_feature_indices), dtype=int)
            if np.any(semantic < 0) or np.any(semantic >= pool.shape[1]):
                raise ValueError("Índices semânticos incompatíveis no motor activo.")
            sem_scale = np.std(ref_train[:, semantic], axis=0) + 1e-9
            sem_center = np.median(ref_train[:, semantic], axis=0)
            sem_distance = np.mean(
                np.abs((pool[:, semantic] - sem_center) / sem_scale), axis=1,
            )
            semantic_gap = sem_distance / (1.0 + sem_distance)
        scores = (
            config.disagreement_weight * disagreement
            + config.uncertainty_weight * entropy
            + config.counterfactual_weight * cf_mask
            + config.c45_disagreement_weight * c45_disagreement
            + config.plausibility_weight * plausibility
            + config.real_label_error_weight * real_label_error
            + config.minority_weight * minority_score
            + config.semantic_gap_weight * semantic_gap
        )
        chosen = _diverse_top_k(
            pool, scores, min(config.budget_per_iteration, len(pool)),
            config.diversity_weight,
        )
        selected_X = pool[chosen]
        selected_y = oracle_labels[chosen]
        selected_w = 1.0 + scores[chosen]
        query_X = np.vstack([query_X, selected_X])
        query_y = np.concatenate([query_y, selected_y])
        query_w = np.concatenate([query_w, selected_w])
        candidate = clone(initial_tree)
        candidate.fit(query_X, query_y, sample_weight=query_w)
        candidate_fidelity = float(accuracy_score(val_labels, candidate.predict(ref_val)))
        candidate_real = real_metrics(val_true, candidate.predict(ref_val))
        candidate_accuracy = None if candidate_real is None else candidate_real["accuracy"]
        if candidate_real is None:
            accepted = candidate_fidelity > best_fidelity + 1e-12
        else:
            accepted = bool(
                candidate_fidelity >= best_fidelity - 0.005
                and candidate_real["balanced_accuracy"]
                >= best_real["balanced_accuracy"] - 0.005
                and candidate_real["macro_f1"] >= best_real["macro_f1"] - 0.005
                and candidate_real["minority_recall"]
                >= best_real["minority_recall"] - 0.01
                and (
                    candidate_fidelity > best_fidelity + 1e-12
                    or candidate_real["balanced_accuracy"]
                    > best_real["balanced_accuracy"] + 1e-12
                    or candidate_real["macro_f1"] > best_real["macro_f1"] + 1e-12
                )
            )
        if accepted:
            best, best_fidelity = candidate, candidate_fidelity
            if candidate_accuracy is not None:
                best_accuracy = candidate_accuracy
                best_real = candidate_real
        trace.append({
            "iteration": iteration + 1,
            "queries": int(len(chosen)),
            "disagreement_rate_pool": float(disagreement.mean()),
            "mean_entropy_pool": float(entropy.mean()),
            "c45_disagreement_rate_pool": float(c45_disagreement.mean()),
            "mean_plausibility_pool": float(plausibility.mean()),
            "real_label_error_proxy_rate": float(real_label_error.mean()),
            "mean_minority_priority": float(minority_score.mean()),
            "semantic_gap_rate": float(semantic_gap.mean()),
            "counterfactual_queries": int(np.count_nonzero(cf_mask[chosen])),
            "mean_counterfactual_plausibility": float(
                cf_mask[chosen][cf_mask[chosen] > 0].mean()
            ) if np.any(cf_mask[chosen] > 0) else None,
            "validation_fidelity": candidate_fidelity,
            "validation_accuracy": candidate_accuracy,
            "validation_balanced_accuracy": (
                None if candidate_real is None else candidate_real["balanced_accuracy"]
            ),
            "validation_macro_f1": (
                None if candidate_real is None else candidate_real["macro_f1"]
            ),
            "validation_minority_recall": (
                None if candidate_real is None else candidate_real["minority_recall"]
            ),
            "accepted": accepted,
        })
    return best, {
        "selection_scope": "internal_training_validation",
        "test_queried": False,
        "initial_fidelity": initial_fidelity,
        "selected_fidelity": best_fidelity,
        "selected_accuracy": best_accuracy,
        "selected_real_label_metrics": best_real,
        "total_queries": int(sum(item["queries"] for item in trace)),
        "trace": trace,
    }, query_X, query_y, query_w
