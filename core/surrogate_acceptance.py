"""Gate de não-inferioridade para a árvore TREPAN Reloaded.

A função deve ser aplicada a um *holdout de aceitação do desenvolvimento*, nunca
ao teste final. Um relatório no teste final pode reutilizar a função apenas com
``selection_role='locked_test_audit_only'`` e não pode alterar o modelo escolhido.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score
from core.classification_metrics import compute_classification_metrics


def _metrics(y_true, y_pred) -> Dict[str, float]:
    return compute_classification_metrics(y_true, y_pred)


def evaluate_surrogate_acceptance_metrics(
    reloaded_metrics: Dict[str, float],
    original_metrics: Dict[str, float],
    *,
    fidelity_to_active_oracle: Optional[float],
    candidate_complexity: Optional[int],
    baseline_complexity: Optional[int],
    tolerance: float = 0.01,
    fidelity_floor: float = 0.90,
    complexity_limit: Optional[int] = None,
    complexity_ratio_limit: float = 1.50,
    oracle_gate_accepted: bool = True,
    c45_metrics: Optional[Dict[str, float]] = None,
    require_c45_noninferiority: bool = False,
    selection_role: str = "development_acceptance_holdout",
) -> Dict[str, Any]:
    tolerance = max(0.0, float(tolerance))
    fidelity_floor = float(fidelity_floor)
    if complexity_limit is None and baseline_complexity is not None:
        complexity_limit = max(
            int(baseline_complexity),
            int(math.ceil(float(baseline_complexity) * float(complexity_ratio_limit))),
        )

    checks = {
        "oracle_gate_accepted": bool(oracle_gate_accepted),
        "precision_macro_noninferior": float(reloaded_metrics.get("precision_macro", 0.0)) >= float(original_metrics.get("precision_macro", 0.0)) - tolerance,
        "recall_macro_noninferior": float(reloaded_metrics.get("recall_macro", 0.0)) >= float(original_metrics.get("recall_macro", 0.0)) - tolerance,
        "macro_f1_noninferior": float(reloaded_metrics.get("macro_f1", 0.0)) >= float(original_metrics.get("macro_f1", 0.0)) - tolerance,
        "balanced_accuracy_noninferior": float(reloaded_metrics.get("balanced_accuracy", 0.0)) >= float(original_metrics.get("balanced_accuracy", 0.0)) - tolerance,
        "accuracy_noninferior": float(reloaded_metrics.get("accuracy", 0.0)) >= float(original_metrics.get("accuracy", 0.0)) - tolerance,
        "fidelity_floor_met": fidelity_to_active_oracle is not None and float(fidelity_to_active_oracle) >= fidelity_floor,
        "complexity_limit_met": (
            True if complexity_limit is None or candidate_complexity is None
            else int(candidate_complexity) <= int(complexity_limit)
        ),
    }
    if require_c45_noninferiority:
        if not c45_metrics:
            checks.update({
                "c45_precision_macro_noninferior": False,
                "c45_recall_macro_noninferior": False,
                "c45_macro_f1_noninferior": False,
                "c45_balanced_accuracy_noninferior": False,
                "c45_accuracy_noninferior": False,
            })
        else:
            checks.update({
                "c45_precision_macro_noninferior": float(reloaded_metrics.get("precision_macro", 0.0)) >= float(c45_metrics.get("precision_macro", 0.0)) - tolerance,
                "c45_recall_macro_noninferior": float(reloaded_metrics.get("recall_macro", 0.0)) >= float(c45_metrics.get("recall_macro", 0.0)) - tolerance,
                "c45_macro_f1_noninferior": float(reloaded_metrics.get("macro_f1", 0.0)) >= float(c45_metrics.get("macro_f1", c45_metrics.get("f1", 0.0))) - tolerance,
                "c45_balanced_accuracy_noninferior": float(reloaded_metrics.get("balanced_accuracy", 0.0)) >= float(c45_metrics.get("balanced_accuracy", 0.0)) - tolerance,
                "c45_accuracy_noninferior": float(reloaded_metrics.get("accuracy", 0.0)) >= float(c45_metrics.get("accuracy", 0.0)) - tolerance,
            })
    c45_checks = {
        name: ok for name, ok in checks.items() if name.startswith("c45_")
    }
    c45_baseline_gate_accepted = (
        None if not require_c45_noninferiority
        else bool(c45_checks) and all(c45_checks.values())
    )
    accepted = bool(all(checks.values()))
    failed = [name for name, ok in checks.items() if not ok]
    return {
        "accepted": accepted,
        "surrogate_gate_accepted": accepted,
        "rejected_by_quality_gate": not accepted,
        "selection_role": selection_role,
        # Nunca usar o teste final para selecionar o modelo. O holdout de
        # desenvolvimento pode decidir o fallback; o locked_test apenas audita.
        "test_used_for_selection": False,
        "development_holdout_used_for_selection": selection_role == "development_acceptance_holdout",
        "tolerance": tolerance,
        "fidelity_floor": fidelity_floor,
        "complexity_ratio_limit": float(complexity_ratio_limit),
        "complexity_limit": complexity_limit,
        "candidate_complexity": candidate_complexity,
        "baseline_complexity": baseline_complexity,
        "reloaded_metrics": dict(reloaded_metrics),
        "trepan_original_metrics": dict(original_metrics),
        "c45_metrics": dict(c45_metrics) if c45_metrics else None,
        "require_c45_noninferiority": bool(require_c45_noninferiority),
        "c45_is_oracle": False,
        "c45_baseline_gate_accepted": c45_baseline_gate_accepted,
        "c45_checks": c45_checks,
        "fidelity_to_active_oracle": None if fidelity_to_active_oracle is None else float(fidelity_to_active_oracle),
        "checks": checks,
        "failed_checks": failed,
        "reason": "accepted_noninferiority_policy" if accepted else "rejected_by_quality_gate:" + ",".join(failed),
        "fallback_required": not accepted and selection_role == "development_acceptance_holdout",
    }


def evaluate_surrogate_acceptance(
    reloaded_model,
    original_model,
    X_reloaded_validation,
    X_original_validation,
    y_validation,
    *,
    active_oracle=None,
    X_active_oracle_validation=None,
    tolerance: float = 0.01,
    fidelity_floor: float = 0.90,
    complexity_ratio_limit: float = 1.50,
    oracle_gate_accepted: bool = True,
    c45_model=None,
    X_c45_validation=None,
    require_c45_noninferiority: bool = False,
    selection_role: str = "development_acceptance_holdout",
) -> Dict[str, Any]:
    y = np.asarray(y_validation)
    reloaded_pred = np.asarray(reloaded_model.predict(X_reloaded_validation))
    original_pred = np.asarray(original_model.predict(X_original_validation))
    active_pred = None
    if active_oracle is not None:
        matrix = X_active_oracle_validation if X_active_oracle_validation is not None else X_reloaded_validation
        active_pred = np.asarray(active_oracle.predict(matrix))

    fidelity = None if active_pred is None else float(accuracy_score(active_pred, reloaded_pred))
    candidate_complexity = int(reloaded_model.get_n_leaves()) if hasattr(reloaded_model, "get_n_leaves") else None
    baseline_complexity = int(original_model.get_n_leaves()) if hasattr(original_model, "get_n_leaves") else None
    c45_metrics = None
    if c45_model is not None:
        c45_matrix = X_c45_validation if X_c45_validation is not None else X_original_validation
        c45_pred = np.asarray(c45_model.predict(c45_matrix))
        c45_metrics = _metrics(y, c45_pred)
    result = evaluate_surrogate_acceptance_metrics(
        _metrics(y, reloaded_pred),
        _metrics(y, original_pred),
        fidelity_to_active_oracle=fidelity,
        candidate_complexity=candidate_complexity,
        baseline_complexity=baseline_complexity,
        tolerance=tolerance,
        fidelity_floor=fidelity_floor,
        complexity_ratio_limit=complexity_ratio_limit,
        oracle_gate_accepted=oracle_gate_accepted,
        c45_metrics=c45_metrics,
        require_c45_noninferiority=require_c45_noninferiority,
        selection_role=selection_role,
    )
    result["n_validation_rows"] = int(len(y))
    return result


__all__ = ["evaluate_surrogate_acceptance", "evaluate_surrogate_acceptance_metrics"]
