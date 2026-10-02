"""Transferência e robustez no único dataset carregado na aplicação."""
from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from counterfactuals.engine import (
    CounterfactualConstraints,
    CounterfactualEngine,
    FeatureSpaceAdapter,
    FeatureSpaceMismatchError,
    OntologyCFValidator,
    _predict,
)


class TransferCategory(str, Enum):
    STRONG_TRANSFER = "STRONG_TRANSFER"
    WEAK_TRANSFER = "WEAK_TRANSFER"
    NO_TRANSFER = "NO_TRANSFER"
    INVALID_MLP_CF = "INVALID_MLP_CF"
    SKIPPED_NO_AGREEMENT = "SKIPPED_NO_AGREEMENT"


def _select_stratified(y: np.ndarray, fraction: float, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    selected = []
    for label in np.unique(y):
        indices = np.where(y == label)[0]
        count = max(1, int(round(len(indices) * fraction))) if len(indices) else 0
        count = min(count, len(indices))
        if count:
            selected.extend(rng.choice(indices, size=count, replace=False).tolist())
    rng.shuffle(selected)
    return np.asarray(selected, dtype=int)


def _space(session: Mapping[str, Any]) -> Dict[str, Any]:
    accepted = bool((session.get("ontology_acceptance") or {}).get("accepted"))
    enriched_ready = (
        accepted
        and session.get("mlp_onto") is not None
        and session.get("X_train_augmented") is not None
        and session.get("feature_names_augmented")
    )
    if enriched_ready:
        return {
            "oracle": session["mlp_onto"],
            "oracle_label": "MLP Ontológica",
            "X": np.asarray(session["X_train_augmented"], dtype=float),
            "y": np.asarray(session.get("y_train_augmented", session["y_train_enc"])),
            "names": list(session["feature_names_augmented"]),
            "ontology_active": True,
        }
    return {
        "oracle": session.get("mlp_original", session["mlp_oracle"]),
        "oracle_label": "MLP Original",
        "X": np.asarray(session.get("X_train_original", session["X_train_enc"]), dtype=float),
        "y": np.asarray(session.get("y_train_original", session["y_train_enc"])),
        "names": list(session.get("feature_names_original", session["transformed_feature_names"])),
        "ontology_active": False,
    }


def _target_names(session: Mapping[str, Any], key: str, fallback: Sequence[str]) -> List[str]:
    value = session.get(key)
    return list(value) if value else list(fallback)


def _adapt_for_model(
    vector: np.ndarray,
    source_names: Sequence[str],
    target_names: Sequence[str],
    *,
    reference: Optional[np.ndarray] = None,
    derive_fn=None,
) -> np.ndarray:
    missing = set(map(str, target_names)) - set(map(str, source_names))
    # Não congele silenciosamente features ontológicas derivadas na linha
    # factual: sem função de derivação, o artigo exige falha controlada.
    safe_reference = None if missing and derive_fn is None else reference
    return FeatureSpaceAdapter.adapt(
        vector,
        source_names,
        target_names,
        reference_target=safe_reference,
        derive_fn=derive_fn,
    )


def _cross_model_robustness(
    oracle: Any,
    tree_a: Any,
    tree_b: Any,
    original: np.ndarray,
    counterfactual: np.ndarray,
    target_class: Any,
    constraints: CounterfactualConstraints,
    source_names: Sequence[str],
    tree_a_names: Sequence[str],
    tree_b_names: Sequence[str],
    *,
    tree_a_reference: Optional[np.ndarray],
    tree_b_reference: Optional[np.ndarray],
    tree_a_derive_fn=None,
    tree_b_derive_fn=None,
    samples: int = 100,
    epsilon: float = 0.02,
    seed: int = 42,
) -> Dict[str, float]:
    rng = np.random.RandomState(seed)
    spans = np.array(
        [max(high - low, 1e-9) for low, high in constraints.feature_ranges.values()]
    )
    mutable = constraints.mutable_indices()
    counts = {"mlp": 0, "trepan": 0, "reloaded": 0, "joint": 0}
    total = max(1, int(samples))
    for _ in range(total):
        perturbed = counterfactual.copy()
        if mutable:
            perturbed[mutable] += rng.normal(0, epsilon, len(mutable)) * spans[mutable]
        perturbed = constraints.project(original, perturbed)
        pred_mlp = _predict(oracle, perturbed)[0]
        vector_a = _adapt_for_model(
            perturbed, source_names, tree_a_names,
            reference=tree_a_reference, derive_fn=tree_a_derive_fn,
        )
        vector_b = _adapt_for_model(
            perturbed, source_names, tree_b_names,
            reference=tree_b_reference, derive_fn=tree_b_derive_fn,
        )
        pred_a = _predict(tree_a, vector_a)[0]
        pred_b = _predict(tree_b, vector_b)[0]
        counts["mlp"] += int(pred_mlp == target_class)
        counts["trepan"] += int(pred_a == target_class)
        counts["reloaded"] += int(pred_b == target_class)
        counts["joint"] += int(pred_mlp == pred_a == pred_b == target_class)
    return {f"{key}_robustness": float(value / total) for key, value in counts.items()}


def evaluate_transfer_protocol(
    session: Mapping[str, Any],
    *,
    methods: Iterable[str] = ("LORE-LOCAL", "CLEAR", "COGS"),
    fraction: float = 0.33,
    seed: int = 42,
    robustness_samples: int = 100,
    robustness_epsilon: float = 0.02,
    progress_fn=None,
    cancel_fn=None,
) -> Dict[str, Any]:
    """Executa P1-P8 exclusivamente na sessão do dataset carregado."""
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("A fracção de amostragem deve estar em ]0, 1].")
    if int(robustness_samples) < 1:
        raise ValueError("robustness_samples deve ser positivo.")
    if float(robustness_epsilon) < 0:
        raise ValueError("robustness_epsilon não pode ser negativo.")
    required = ("tree_a", "tree_b", "mlp_oracle", "X_train_enc", "y_train_enc")
    missing = [key for key in required if session.get(key) is None]
    if missing:
        raise ValueError("Sessão incompleta para P1-P8: " + ", ".join(missing))

    def progress(stage: str, pct: int, message: str) -> None:
        if progress_fn:
            progress_fn(stage, pct, message)

    def cancelled() -> bool:
        return bool(cancel_fn and cancel_fn())

    space = _space(session)
    oracle = space["oracle"]
    X = space["X"]
    y = space["y"]
    source_names = space["names"]
    tree_a = session["tree_a"]
    tree_b = session["tree_b"]
    tree_a_names = _target_names(session, "tree_a_feature_names", source_names)
    tree_b_names = _target_names(session, "tree_b_feature_names", source_names)
    X_a = session.get("X_tree_a_reference")
    X_b = session.get("X_tree_b_reference")
    if X_a is not None:
        X_a = np.asarray(X_a, dtype=float)
    if X_b is not None:
        X_b = np.asarray(X_b, dtype=float)

    config = dict(session.get("constraints") or session.get("config") or {})
    constraints = CounterfactualConstraints.from_data(X, source_names, config)
    ontology_validator = OntologyCFValidator(
        session.get("ontology") if space["ontology_active"] else None,
        session.get("ontology_feature_mapping"),
        session.get("semantic_rules"),
    )
    engine = CounterfactualEngine(
        oracle,
        X,
        source_names,
        y_reference=y,
        constraints=constraints,
        ontology_validator=ontology_validator,
        seed=seed,
    )
    selected = _select_stratified(y, fraction, seed)
    method_list = [str(method).upper().replace("_", "-") for method in methods]
    rows: List[Dict[str, Any]] = []
    progress("p4", 5, f"P4: {len(selected)} instâncias seleccionadas")

    for position, index in enumerate(selected):
        if cancelled():
            raise InterruptedError("Avaliação de transferência cancelada")
        original = X[index]
        ref_a = X_a[index] if X_a is not None and index < len(X_a) else None
        ref_b = X_b[index] if X_b is not None and index < len(X_b) else None
        vector_a = _adapt_for_model(
            original, source_names, tree_a_names,
            reference=ref_a, derive_fn=session.get("tree_a_derive_fn"),
        )
        vector_b = _adapt_for_model(
            original, source_names, tree_b_names,
            reference=ref_b, derive_fn=session.get("tree_b_derive_fn"),
        )
        factual_mlp = _predict(oracle, original)[0]
        factual_a = _predict(tree_a, vector_a)[0]
        factual_b = _predict(tree_b, vector_b)[0]
        agreement = bool(factual_mlp == factual_a == factual_b)

        for method in method_list:
            base_row = {
                "instance_index": int(index),
                "method": method,
                "factual_mlp": _scalar(factual_mlp),
                "factual_trepan": _scalar(factual_a),
                "factual_reloaded": _scalar(factual_b),
                "initial_agreement": agreement,
                "ontology_active": bool(space["ontology_active"]),
                "oracle_label": space["oracle_label"],
            }
            if not agreement:
                rows.append({**base_row, "category": TransferCategory.SKIPPED_NO_AGREEMENT.value})
                continue

            result = engine.generate(
                original,
                method=method,
                model_type="mlp",
                total_cfs=1,
                robustness_samples=max(10, robustness_samples // 2),
                robustness_epsilon=robustness_epsilon,
            )
            best = result.get("best_candidate")
            if best is None or not best["metrics"]["validity"]:
                rows.append({
                    **base_row,
                    "category": TransferCategory.INVALID_MLP_CF.value,
                    "cf_valid_mlp": False,
                })
                continue
            cf = np.asarray(best["vector"], dtype=float)
            cf_class_mlp = _predict(oracle, cf)[0]
            cf_a = _adapt_for_model(
                cf, source_names, tree_a_names,
                reference=ref_a, derive_fn=session.get("tree_a_derive_fn"),
            )
            cf_b = _adapt_for_model(
                cf, source_names, tree_b_names,
                reference=ref_b, derive_fn=session.get("tree_b_derive_fn"),
            )
            cf_class_a = _predict(tree_a, cf_a)[0]
            cf_class_b = _predict(tree_b, cf_b)[0]
            changed_a = cf_class_a != factual_a
            changed_b = cf_class_b != factual_b
            if changed_a and changed_b and cf_class_a == cf_class_b == cf_class_mlp:
                category = TransferCategory.STRONG_TRANSFER
            elif changed_a and changed_b:
                category = TransferCategory.WEAK_TRANSFER
            else:
                category = TransferCategory.NO_TRANSFER
            robustness = _cross_model_robustness(
                oracle,
                tree_a,
                tree_b,
                original,
                cf,
                cf_class_mlp,
                constraints,
                source_names,
                tree_a_names,
                tree_b_names,
                tree_a_reference=ref_a,
                tree_b_reference=ref_b,
                tree_a_derive_fn=session.get("tree_a_derive_fn"),
                tree_b_derive_fn=session.get("tree_b_derive_fn"),
                samples=robustness_samples,
                epsilon=robustness_epsilon,
                seed=seed + int(index),
            )
            rows.append({
                **base_row,
                "category": category.value,
                "cf_valid_mlp": True,
                "cf_class_mlp": _scalar(cf_class_mlp),
                "cf_class_trepan": _scalar(cf_class_a),
                "cf_class_reloaded": _scalar(cf_class_b),
                "trepan_changed": bool(changed_a),
                "reloaded_changed": bool(changed_b),
                "trepan_aligned": bool(cf_class_a == cf_class_mlp),
                "reloaded_aligned": bool(cf_class_b == cf_class_mlp),
                "counterfactual": cf.tolist(),
                "sparsity": best["metrics"]["sparsity"],
                "proximity": best["metrics"]["proximity"],
                "robustness": best["metrics"]["robustness"],
                **robustness,
            })
        progress(
            "p7",
            5 + int(90 * (position + 1) / max(len(selected), 1)),
            f"P5-P7: instância {position + 1}/{len(selected)}",
        )

    summary = aggregate_transfer_metrics(rows)
    summary_by_method = {
        method: aggregate_transfer_metrics(
            [row for row in rows if row.get("method") == method]
        )
        for method in method_list
    }
    progress("done", 100, "P1-P8 concluídos")
    return {
        "status": "success",
        "protocol": "P1-P8",
        "dataset": session.get("dataset_name", "session"),
        "seed": int(seed),
        "fraction": float(fraction),
        "selected_indices": selected.tolist(),
        "methods": method_list,
        "oracle_label": space["oracle_label"],
        "ontology_active": bool(space["ontology_active"]),
        "rows": rows,
        "summary": summary,
        "summary_by_method": summary_by_method,
        "interpretation": _interpret(summary),
    }


def aggregate_transfer_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    categories = Counter(row.get("category") for row in rows)
    skipped = categories[TransferCategory.SKIPPED_NO_AGREEMENT.value]
    eligible = [
        row for row in rows
        if row.get("category") != TransferCategory.SKIPPED_NO_AGREEMENT.value
    ]
    valid = [row for row in eligible if row.get("cf_valid_mlp")]
    denom_eligible = max(len(eligible), 1)
    denom_valid = max(len(valid), 1)

    def rate(predicate, records=valid, denominator=denom_valid):
        return float(sum(1 for row in records if predicate(row)) / denominator) if records else 0.0

    robustness_keys = (
        "robustness", "mlp_robustness", "trepan_robustness",
        "reloaded_robustness", "joint_robustness",
    )
    summary = {
        "total_pairs": total,
        "eligible_pairs": len(eligible),
        "skipped_no_agreement": skipped,
        "initial_agreement_rate": float((total - skipped) / total) if total else 0.0,
        "valid_mlp_cf_rate": float(len(valid) / denom_eligible) if eligible else 0.0,
        "strong_transfer_rate": rate(lambda row: row.get("category") == TransferCategory.STRONG_TRANSFER.value),
        "weak_transfer_rate": rate(lambda row: row.get("category") == TransferCategory.WEAK_TRANSFER.value),
        "no_transfer_rate": rate(lambda row: row.get("category") == TransferCategory.NO_TRANSFER.value),
        "trepan_transfer_rate": rate(lambda row: row.get("trepan_changed")),
        "reloaded_transfer_rate": rate(lambda row: row.get("reloaded_changed")),
        "trepan_alignment_rate": rate(lambda row: row.get("trepan_aligned")),
        "reloaded_alignment_rate": rate(lambda row: row.get("reloaded_aligned")),
        "category_counts": dict(categories),
    }
    for key in robustness_keys:
        values = [float(row[key]) for row in valid if row.get(key) is not None]
        summary[f"mean_{key}"] = float(np.mean(values)) if values else None
    return summary


def _interpret(summary: Mapping[str, Any]) -> str:
    strong = float(summary.get("strong_transfer_rate") or 0.0)
    no_transfer = float(summary.get("no_transfer_rate") or 0.0)
    original = float(summary.get("trepan_transfer_rate") or 0.0)
    reloaded = float(summary.get("reloaded_transfer_rate") or 0.0)
    if strong >= 0.75:
        base = "Transferência forte elevada: as árvores preservam bem o comportamento contrafactual."
    elif no_transfer >= 0.5:
        base = "Sem transferência elevada: as fronteiras locais da MLP não são bem preservadas."
    else:
        base = "Transferência contrafactual mista: examine categorias e robustez por método."
    if reloaded > original:
        return base + " O Trepan Reloaded supera o Trepan Original neste protocolo."
    if original > reloaded:
        return base + " O Trepan Original supera o Reloaded; reveja espaço e enriquecimento ontológico."
    return base + " As duas árvores apresentam a mesma taxa de mudança."


def _scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


__all__ = [
    "TransferCategory",
    "FeatureSpaceMismatchError",
    "evaluate_transfer_protocol",
    "aggregate_transfer_metrics",
]
