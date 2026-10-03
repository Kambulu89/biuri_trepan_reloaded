"""Árvore explicativa local enriquecida por contrafactuais válidos."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from counterfactuals.global_rules import extract_global_rules, unwrap_tree


def _predict(model: Any, values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    return np.asarray(model.predict(array)).reshape(-1)


def _compatible_tree(model: Any, n_features: int) -> Optional[Any]:
    try:
        tree = unwrap_tree(model)
    except (TypeError, AttributeError):
        return None
    return tree if int(getattr(tree, "n_features_in_", n_features)) == n_features else None


def _model_fingerprint(model: Any, X: np.ndarray, feature_names: Sequence[str]) -> str:
    sample_idx = np.linspace(0, len(X) - 1, min(31, len(X))).astype(int)
    payload = {
        "class": f"{model.__class__.__module__}.{model.__class__.__qualname__}",
        "features": list(map(str, feature_names)),
        "predictions": _predict(model, X[sample_idx]).tolist(),
        "params": (
            model.get_params(deep=False) if hasattr(model, "get_params") else {}
        ),
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _candidate_plausibility(item: Mapping[str, Any]) -> float:
    metrics = item.get("metrics") or {}
    if not bool(metrics.get("validity")):
        return 0.0
    values = []
    for key in ("plausibility", "proximity", "robustness", "ontology_coherence"):
        value = metrics.get(key)
        if value is not None and np.isfinite(float(value)):
            values.append(float(np.clip(value, 0.0, 1.0)))
    return float(np.mean(values)) if values else 1.0


def build_counterfactual_tree(
    oracle: Any,
    X_reference: Any,
    generation_result: Mapping[str, Any],
    feature_names: Sequence[str],
    *,
    original_tree: Any = None,
    class_labels: Optional[Mapping[Any, Any]] = None,
    max_depth: int = 5,
    neighborhood_size: int = 300,
    seed: int = 42,
    model_name: str = "modelo",
    dataset_name: str = "dataset_carregado",
) -> Dict[str, Any]:
    """Constrói a *árvore local de contrafactuais* (CF-LocalTree).

    Finalidade: resumir, em regras legíveis, a fronteira de decisão do modelo
    explicado na vizinhança da instância activa, usando o factual e os CFs
    validados como sementes. NÃO é a árvore global TREPAN Original/Reloaded:
    é um substituto local, treinado com o motor TREPAN histórico apenas como
    algoritmo de indução, e a sua fidelidade refere-se ao oráculo local.
    O algoritmo TREPAN não usa pesos de amostra, pelo que nenhuma ponderação
    de CFs é aplicada (as sementes entram com peso uniforme).

    A implementação legada treinava por vezes com uma amostra repetida ou
    rótulos contraditórios. Esta versão reserva uma vizinhança para avaliação,
    rotula dados reais pelo oráculo e dá peso adicional apenas a CFs válidos.
    """
    from sklearn.model_selection import train_test_split
    from core.trepan_original import TrepanOriginalClassifier

    X = np.asarray(X_reference, dtype=float)
    names = [str(name) for name in feature_names]
    original = np.asarray(generation_result.get("original_instance"), dtype=float).reshape(-1)
    if X.ndim != 2 or X.shape[1] != len(names) or len(original) != len(names):
        raise ValueError("O resultado contrafactual não pertence ao espaço do dataset activo.")
    result_names = list(generation_result.get("feature_names") or [])
    if result_names and result_names != names:
        raise ValueError("As features do resultado contrafactual não correspondem ao modelo activo.")

    valid_candidates = [
        item for item in generation_result.get("candidates") or []
        if bool((item.get("metrics") or {}).get("validity"))
    ]
    if not valid_candidates:
        raise ValueError("É necessário gerar pelo menos um contrafactual válido antes da árvore CF.")
    cf_vectors = np.asarray([item["vector"] for item in valid_candidates], dtype=float)
    cf_plausibility = np.asarray([
        _candidate_plausibility(item) for item in valid_candidates
    ], dtype=float)
    if cf_vectors.ndim != 2 or cf_vectors.shape[1] != len(names):
        raise ValueError("Um candidato válido está desalinhado com as features activas.")

    scale = np.std(X, axis=0)
    scale = np.where(scale < 1e-10, 1.0, scale)
    distance = np.linalg.norm((X - original) / scale, axis=1)
    count = min(len(X), max(20, int(neighborhood_size)))
    neighbors = X[np.argsort(distance)[:count]]
    neighbor_labels = _predict(oracle, neighbors)

    evaluation_mode = "holdout"
    unique, class_counts = np.unique(neighbor_labels, return_counts=True)
    if len(neighbors) >= 20 and len(unique) >= 2 and int(np.min(class_counts)) >= 2:
        X_train, X_eval, y_train, y_eval = train_test_split(
            neighbors,
            neighbor_labels,
            test_size=0.30,
            random_state=seed,
            stratify=neighbor_labels,
        )
    else:
        X_train, y_train = neighbors, neighbor_labels
        X_eval, y_eval = neighbors, neighbor_labels
        evaluation_mode = "resubstitution_insufficient_local_classes"

    factual_label = _predict(oracle, original)[0]
    cf_labels = _predict(oracle, cf_vectors)
    augmented_X = np.vstack([X_train, original.reshape(1, -1), cf_vectors])
    augmented_y = np.concatenate([y_train, np.asarray([factual_label]), cf_labels])
    model_fingerprint = _model_fingerprint(oracle, X, names)
    effective_seed = int(
        (int(model_fingerprint[:8], 16) ^ int(seed)) % (2**31 - 1)
    )
    # Produção: a árvore local pertence à mesma família TREPAN histórica do
    # restante sistema. Contrafactuais válidos são sementes adicionais, mas
    # os rótulos continuam a ser consultados no próprio oráculo.
    local_depth = max(1, int(max_depth))
    local_nodes = min(31, max(3, 2 ** min(local_depth, 5) - 1))
    local_min_sample = max(len(augmented_X), min(300, max(60, len(augmented_X) * 2)))
    tree = TrepanOriginalClassifier(
        max_nodes=local_nodes,
        max_depth=local_depth,
        min_samples_leaf=2 if len(augmented_X) >= 10 else 1,
        min_sample=local_min_sample,
        max_n=3,
        beam_width=2,
        max_features_per_node=min(12, len(names)),
        max_queries=max(300, local_min_sample * 2),
        random_state=effective_seed,
    )
    tree.fit(augmented_X, oracle=oracle, feature_names=names)

    eval_prediction = tree.predict(X_eval)
    original_reference = _compatible_tree(original_tree, len(names))
    metrics: Dict[str, Any] = {
        "fidelity_to_oracle": float(np.mean(eval_prediction == y_eval)),
        "counterfactual_fidelity": float(np.mean(tree.predict(cf_vectors) == cf_labels)),
        "factual_fidelity": float(tree.predict(original.reshape(1, -1))[0] == factual_label),
        "depth": int(tree.get_depth()),
        "n_nodes": int(getattr(tree, "node_count_", 0)),
        "n_leaves": int(tree.get_n_leaves()),
        "n_reference_neighbors": int(len(neighbors)),
        "n_training_neighbors": int(len(X_train)),
        "n_evaluation_neighbors": int(len(X_eval)),
        "n_valid_counterfactuals": int(len(cf_vectors)),
        "evaluation_mode": evaluation_mode,
        "mean_cf_plausibility": float(cf_plausibility.mean()),
    }
    if original_reference is not None:
        metrics["fidelity_to_original_tree"] = float(
            np.mean(eval_prediction == original_reference.predict(X_eval))
        )
    else:
        metrics["fidelity_to_original_tree"] = None

    labels = dict(class_labels or {})
    rules = extract_global_rules(tree, names, labels)
    tree_rules = [{
        "rule_id": rule.rule_id,
        "predicted_class": rule.predicted_class,
        "predicted_class_name": rule.predicted_class_name,
        "support": rule.support,
        "confidence": rule.confidence,
        "conditions": [condition.text() for condition in rule.conditions],
        "rule": rule.text(),
    } for rule in rules]
    narrative = (
        f"A árvore explicativa contrafactual foi construída para {model_name} com "
        f"{len(neighbors)} vizinhos reais e {len(cf_vectors)} contrafactuais válidos. "
        f"A fidelidade local ao oráculo é {metrics['fidelity_to_oracle']:.1%}; "
        f"a avaliação usada foi {evaluation_mode}."
    )
    tree_signature = hashlib.sha256(
        json.dumps(tree.export_rules(), sort_keys=True, default=str).encode("utf-8")
        + model_fingerprint.encode("ascii")
    ).hexdigest()
    return {
        "result_type": "counterfactual_tree",
        "tree_kind": "CF-LocalTree",
        "tree_kind_note": "Árvore local de contrafactuais; distinta de TREPAN Original/Reloaded globais.",
        "scope": "loaded_dataset_only",
        "dataset": dataset_name,
        "target_model": model_name,
        "oracle_fingerprint": model_fingerprint,
        "tree_fingerprint": tree_signature,
        "effective_seed": effective_seed,
        "source_method": generation_result.get("method"),
        "feature_names": names,
        "class_names": [str(labels.get(value, value)) for value in tree.classes_],
        "tree_rules": tree_rules,
        "aggregate_metrics": metrics,
        "narrative": narrative,
        "methodology": {
            "training": "TREPAN histórico local: vizinhança real + factual + CFs válidos como sementes; rótulos consultados no oráculo",
            "evaluation": evaluation_mode,
            "non_mutating": True,
            "oracle_isolated_per_target": True,
            "cf_weighting": "none (TREPAN ignora sample_weight; sementes com peso uniforme)",
            "purpose": "resumo local da fronteira de decisão do modelo explicado; não é explicação causal",
            "causality": "NOT_CLAIMED",
        },
        "_runtime_tree_model": tree,
    }


__all__ = ["build_counterfactual_tree"]
