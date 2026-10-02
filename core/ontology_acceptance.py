"""Gate científico do MLP Ontológico.

A aceitação nunca depende apenas de utility composta. Precision Macro, Recall
Macro, Macro-F1 e Balanced Accuracy devem ser não-inferiores dentro da margem
pré-definida; Accuracy pode ser incluída como métrica primária opcional.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from core.classification_metrics import compute_classification_metrics


def compute_oracle_test_metrics(model, X_test, y_test) -> Dict[str, Any]:
    y_pred = model.predict(X_test)
    out = compute_classification_metrics(y_test, y_pred)
    # aliases de compatibilidade com código histórico
    out["precision"] = out["precision_weighted"]
    out["f1"] = out["macro_f1"]
    out["recall"] = out["recall_macro"]
    return out


def classify_ontology_impact(precision_macro_gain: float) -> str:
    if precision_macro_gain > 0:
        return "beneficial"
    if precision_macro_gain == 0:
        return "neutral"
    return "harmful"


def evaluate_ontological_oracle_acceptance(
    original_metrics: Dict[str, float],
    ontological_metrics: Dict[str, float],
    *,
    tolerance: float = 0.01,
    require_accuracy: bool = True,
) -> Dict[str, Any]:
    tolerance = max(0.0, float(tolerance))
    aliases = {
        "precision_macro": ("precision_macro", "precision"),
        "recall_macro": ("recall_macro", "recall"),
        "macro_f1": ("macro_f1", "f1"),
        "balanced_accuracy": ("balanced_accuracy",),
        "accuracy": ("accuracy",),
    }

    def value(block, canonical):
        for key in aliases[canonical]:
            if block.get(key) is not None:
                return float(block[key])
        raise KeyError(f"Métrica obrigatória ausente: {canonical}")

    required = ["precision_macro", "recall_macro", "macro_f1", "balanced_accuracy"]
    if require_accuracy:
        required.append("accuracy")
    checks: Dict[str, bool] = {}
    deltas: Dict[str, float] = {}
    normalized_original: Dict[str, float] = {}
    normalized_onto: Dict[str, float] = {}
    for metric in required:
        base = value(original_metrics, metric)
        candidate = value(ontological_metrics, metric)
        normalized_original[metric] = base
        normalized_onto[metric] = candidate
        checks[f"{metric}_noninferior"] = candidate >= base - tolerance
        deltas[f"{metric}_gain"] = candidate - base

    accepted = bool(all(checks.values()))
    return {
        "accepted": accepted,
        "oracle_gate_accepted": accepted,
        "fallback_to_original": not accepted,
        "active_oracle": "MLP Ontológico" if accepted else "MLP Original",
        "tolerance": tolerance,
        "require_accuracy": bool(require_accuracy),
        "checks": checks,
        "failed_checks": [k for k, ok in checks.items() if not ok],
        "original_metrics": normalized_original,
        "ontological_metrics": normalized_onto,
        **deltas,
        "ontology_impact": classify_ontology_impact(deltas["precision_macro_gain"]),
        "reason": "STRICT_NONINFERIORITY_PASSED" if accepted else "STRICT_NONINFERIORITY_FAILED:" + ",".join(k for k, ok in checks.items() if not ok),
        "acceptance_rule": "precision_macro+recall_macro+macro_f1+balanced_accuracy" + ("+accuracy" if require_accuracy else ""),
        "ontology_accepted_label": "Sim" if accepted else "Não",
    }


def evaluate_ontology_acceptance(
    accuracy_original: float,
    accuracy_onto: float,
    f1_original: float,
    f1_onto: float,
    balanced_accuracy_original: Optional[float] = None,
    balanced_accuracy_onto: Optional[float] = None,
    tolerance: float = 0.01,
    precision_macro_original: Optional[float] = None,
    precision_macro_onto: Optional[float] = None,
    recall_macro_original: Optional[float] = None,
    recall_macro_onto: Optional[float] = None,
) -> Dict[str, Any]:
    """API compatível com V9.1, agora aplicando o contrato científico completo.

    Chamadores antigos que não fornecem Precision/Recall usam Macro-F1 como
    fallback conservador de compatibilidade. O fluxo novo deve fornecer todas.
    """
    original = {
        "accuracy": accuracy_original,
        "macro_f1": f1_original,
        "balanced_accuracy": balanced_accuracy_original if balanced_accuracy_original is not None else accuracy_original,
        "precision_macro": precision_macro_original if precision_macro_original is not None else f1_original,
        "recall_macro": recall_macro_original if recall_macro_original is not None else f1_original,
    }
    onto = {
        "accuracy": accuracy_onto,
        "macro_f1": f1_onto,
        "balanced_accuracy": balanced_accuracy_onto if balanced_accuracy_onto is not None else accuracy_onto,
        "precision_macro": precision_macro_onto if precision_macro_onto is not None else f1_onto,
        "recall_macro": recall_macro_onto if recall_macro_onto is not None else f1_onto,
    }
    result = evaluate_ontological_oracle_acceptance(original, onto, tolerance=tolerance, require_accuracy=True)
    # aliases históricos esperados pelos testes/GUI
    result.update({
        "accuracy_original": accuracy_original,
        "accuracy_onto": accuracy_onto,
        "f1_original": f1_original,
        "f1_onto": f1_onto,
        "balanced_accuracy_original": balanced_accuracy_original,
        "balanced_accuracy_onto": balanced_accuracy_onto,
        "balanced_accuracy_gain": None if balanced_accuracy_original is None or balanced_accuracy_onto is None else balanced_accuracy_onto - balanced_accuracy_original,
        "accuracy_gain": accuracy_onto - accuracy_original,
        "f1_gain": f1_onto - f1_original,
        "accuracy_accepted": result["checks"].get("accuracy_noninferior", True),
        "f1_accepted": result["checks"].get("macro_f1_noninferior", True),
        "balanced_accepted": result["checks"].get("balanced_accuracy_noninferior", True),
        "primary_accepted": result["accepted"],
        "f1_role": "macro_f1",
    })
    return result


def evaluate_selected_teacher_candidate(
    *,
    original_metrics: Dict[str, float],
    selected_metrics: Dict[str, float],
    hybrid_weights: Dict[str, float],
    tolerance: float = 0.01,
    ontology_feature_gate_accepted: bool = True,
) -> Dict[str, Any]:
    # Compatibilidade com resultados legados que ainda só trazem macro_f1/BA/accuracy.
    # O pipeline novo fornece Precision/Recall explícitas; aqui o fallback evita quebrar
    # artefactos antigos, sem alterar a regra quando as métricas estão presentes.
    original_for_gate = dict(original_metrics)
    selected_for_gate = dict(selected_metrics)
    for block in (original_for_gate, selected_for_gate):
        if block.get("precision_macro") is None:
            block["precision_macro"] = block.get("macro_f1", block.get("f1", 0.0))
        if block.get("recall_macro") is None:
            block["recall_macro"] = block.get("macro_f1", block.get("f1", 0.0))
    strict = evaluate_ontological_oracle_acceptance(
        original_for_gate, selected_for_gate, tolerance=tolerance, require_accuracy=True
    )
    strict.setdefault("accuracy_accepted", strict.get("checks", {}).get("accuracy_noninferior", True))
    strict.setdefault("f1_accepted", strict.get("checks", {}).get("macro_f1_noninferior", True))
    strict.setdefault("balanced_accepted", strict.get("checks", {}).get("balanced_accuracy_noninferior", True))
    strict.setdefault("precision_macro_accepted", strict.get("checks", {}).get("precision_macro_noninferior", True))
    strict.setdefault("recall_macro_accepted", strict.get("checks", {}).get("recall_macro_noninferior", True))
    ontology_mass = float(hybrid_weights.get("ontological", 0.0)) + float(hybrid_weights.get("residual", 0.0))
    has_ontology_mass = ontology_mass > 0.0
    accepted = bool(ontology_feature_gate_accepted and has_ontology_mass and strict.get("accepted"))
    if not ontology_feature_gate_accepted:
        reason = "ONTOLOGY_FEATURE_GATE_REJECTED"
    elif not has_ontology_mass:
        reason = "ZERO_ONTOLOGY_MASS_IN_SELECTED_HYBRID"
    elif not strict.get("accepted"):
        reason = "STRICT_NONINFERIORITY_FAILED"
    else:
        reason = "STRICT_NONINFERIORITY_PASSED"
    return {
        **strict,
        "accepted": accepted,
        "oracle_gate_accepted": accepted,
        "ontology_feature_gate_accepted": bool(ontology_feature_gate_accepted),
        "candidate_teacher_has_ontology": has_ontology_mass,
        "ontology_mass": ontology_mass,
        "reason": reason,
        "ontology_accepted_label": "Sim" if accepted else "Não",
        "fallback_to_original": not accepted,
        "active_oracle": "MLP Ontológico/Híbrido" if accepted else "MLP Original",
    }


def select_oracle_model(mlp_original, mlp_residual_oracle, acceptance: Optional[Dict[str, Any]]) -> Tuple[Any, str]:
    if acceptance and acceptance.get("accepted") and mlp_residual_oracle is not None:
        return mlp_residual_oracle, "MLP Residual Ontológico"
    return mlp_original, "MLP Original"


__all__ = [
    "compute_oracle_test_metrics", "evaluate_ontology_acceptance",
    "evaluate_ontological_oracle_acceptance", "evaluate_selected_teacher_candidate",
    "select_oracle_model",
]
