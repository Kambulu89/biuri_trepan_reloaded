"""Gate auditável do baseline C4.5 para TREPAN.

O C4.5 é um *baseline supervisionado pelos rótulos reais*, nunca um oráculo do
TREPAN. Este módulo não altera predições nem métricas: apenas declara se uma
árvore TREPAN alcançou o baseline no mesmo conjunto de avaliação.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional


DEFAULT_PRIMARY_METRICS = (
    "precision_macro",
    "recall_macro",
    "macro_f1",
    "balanced_accuracy",
)


def _value(metrics: Dict[str, Any], key: str) -> float:
    aliases = {
        "macro_f1": ("macro_f1", "f1_macro", "f1"),
        "recall_macro": ("recall_macro", "recall"),
        "precision_macro": ("precision_macro", "precision"),
        "balanced_accuracy": ("balanced_accuracy",),
        "accuracy": ("accuracy",),
    }
    for candidate in aliases.get(key, (key,)):
        value = metrics.get(candidate)
        if value is not None:
            return float(value)
    return 0.0


def evaluate_c45_baseline_gate(
    trepan_metrics: Dict[str, Any],
    c45_metrics: Dict[str, Any],
    *,
    model_label: str,
    tolerance: float = 0.0,
    primary_metrics: Iterable[str] = DEFAULT_PRIMARY_METRICS,
    include_accuracy: bool = True,
    selection_role: str = "reporting_only",
) -> Dict[str, Any]:
    """Compara TREPAN e C4.5 no mesmo holdout, sem transformar C4.5 em oráculo.

    ``accepted`` significa apenas "alcançou o baseline C4.5 dentro da margem".
    Em ``locked_test_audit_only`` o resultado é exclusivamente descritivo e
    nunca pode alterar o modelo selecionado.
    """
    tolerance = max(0.0, float(tolerance))
    metric_names = list(primary_metrics)
    if include_accuracy and "accuracy" not in metric_names:
        metric_names.append("accuracy")

    checks: Dict[str, bool] = {}
    deltas: Dict[str, float] = {}
    values: Dict[str, Dict[str, float]] = {}
    for metric in metric_names:
        trepan_value = _value(trepan_metrics, metric)
        c45_value = _value(c45_metrics, metric)
        checks[f"{metric}_vs_c45_noninferior"] = trepan_value >= c45_value - tolerance
        deltas[metric] = trepan_value - c45_value
        values[metric] = {"trepan": trepan_value, "c45": c45_value}

    accepted = bool(checks) and all(checks.values())
    return {
        "model_label": str(model_label),
        "accepted": accepted,
        "c45_baseline_gate_accepted": accepted,
        "rejected_by_c45_baseline_gate": not accepted,
        "selection_role": selection_role,
        "test_used_for_selection": False,
        "c45_is_oracle": False,
        "comparison_reference": "real_labels_same_evaluation_rows",
        "tolerance": tolerance,
        "checks": checks,
        "deltas": deltas,
        "values": values,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "reason": (
            "c45_baseline_reached"
            if accepted
            else "c45_baseline_not_reached:" + ",".join(name for name, ok in checks.items() if not ok)
        ),
    }


def evaluate_c45_baseline_pair(
    trepan_original_metrics: Optional[Dict[str, Any]],
    trepan_reloaded_metrics: Optional[Dict[str, Any]],
    c45_metrics: Optional[Dict[str, Any]],
    *,
    tolerance: float = 0.0,
    selection_role: str = "reporting_only",
) -> Dict[str, Any]:
    """Avalia Original e Reloaded contra o mesmo baseline C4.5."""
    if not c45_metrics:
        return {
            "available": False,
            "c45_is_oracle": False,
            "reason": "c45_metrics_unavailable",
            "trepan_original": None,
            "trepan_reloaded": None,
        }

    def run(metrics, label):
        if not metrics:
            return None
        return evaluate_c45_baseline_gate(
            metrics,
            c45_metrics,
            model_label=label,
            tolerance=tolerance,
            selection_role=selection_role,
        )

    original = run(trepan_original_metrics, "TREPAN Original")
    reloaded = run(trepan_reloaded_metrics, "TREPAN Reloaded")
    return {
        "available": True,
        "c45_is_oracle": False,
        "baseline_label": "C4.5-Nativo",
        "selection_role": selection_role,
        "tolerance": max(0.0, float(tolerance)),
        "trepan_original": original,
        "trepan_reloaded": reloaded,
        "all_trepan_reach_c45_baseline": bool(
            original and reloaded and original.get("accepted") and reloaded.get("accepted")
        ),
    }


__all__ = [
    "DEFAULT_PRIMARY_METRICS",
    "evaluate_c45_baseline_gate",
    "evaluate_c45_baseline_pair",
]
