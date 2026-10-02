"""Apresentação textual, sem dependência de Qt, do estado ontológico."""
from __future__ import annotations

from typing import Any, Dict, Optional

from core.ontology_stage_status import build_ontology_stage_status


def _pct(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.1f}%".replace(".", ",")
    except (TypeError, ValueError):
        return "N/A"


def _num(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "N/A"


def build_ontology_status_text(
    quality_report: Optional[Dict[str, Any]],
    acceptance: Optional[Dict[str, Any]],
    *,
    selected_oracle_label: str = "MLP Original",
) -> str:
    acc = dict(acceptance or {})
    state = build_ontology_stage_status(quality_report, acc)
    mapped = state["mapped_features"]
    total = state["total_features"]
    coverage = state["feature_coverage"]

    quality_label = "VÁLIDA" if state["ontology_quality_accepted"] else "REJEITADA"
    feature_label = (
        "ACEITE" if state["ontology_feature_engineering_accepted"] else "NÃO ACEITE"
    )
    trepan_label = "DISPONÍVEL" if state["trepan_semantic_use_allowed"] else "INDISPONÍVEL"
    oracle_label = "ACEITE" if state["oracle_gate_accepted"] else "NÃO ACEITE"

    lines = [
        "🔬 Validação ontológica e do professor (apenas desenvolvimento):",
        f"   Ontologia estrutural: {quality_label} [{state['ontology_quality_status']}]",
        f"   Mapeamento OWL: {mapped}/{total} ({_pct(coverage)})",
        f"   Enriquecimento de features para o MLP: {feature_label}",
        f"   Gate de utilidade OOF: {state['semantic_utility_status']}",
        f"   Gate do professor ontológico: {oracle_label}",
        f"   Uso semântico no TREPAN Reloaded: {trepan_label}",
        f"   Accuracy MLP Original: {_pct(acc.get('accuracy_original'))}",
        f"   Accuracy professor selecionado: {_pct(acc.get('accuracy_onto'))}",
        f"   Balanced Accuracy MLP Original: {_pct(acc.get('balanced_accuracy_original'))}",
        f"   Balanced Accuracy professor selecionado: {_pct(acc.get('balanced_accuracy_onto'))}",
        f"   Macro-F1 MLP Original: {_pct(acc.get('f1_original'))}",
        f"   Macro-F1 professor selecionado: {_pct(acc.get('f1_onto'))}",
        f"   Ganho Balanced Accuracy: {_pct(acc.get('balanced_accuracy_gain'))}",
        f"   Professor contém sinal OWL: {'Sim' if state['teacher_has_ontology'] else 'Não'}",
        f"   Oráculo selecionado: {selected_oracle_label or 'MLP Original'}",
    ]
    semantic = acc.get("feature_selection") or {}
    baseline = semantic.get("baseline") or {}
    candidate = semantic.get("selected_candidate") or {}
    if semantic:
        gate_estimator = semantic.get("gate_estimator") or {}
        fold_refit = semantic.get("fold_local_semantic_refit") or {}
        candidate_names = list(candidate.get("semantic_names") or [])
        preview = ", ".join(candidate_names[:6])
        if len(candidate_names) > 6:
            preview += f", ... (+{len(candidate_names) - 6})"
        lines.extend([
            "   --- Diagnóstico do gate semântico ---",
            f"   Modelo do gate: {gate_estimator.get('family', 'N/A')} "
            f"[{gate_estimator.get('source', 'N/A')}]",
            f"   Refit das estatísticas OWL em cada fold: "
            f"{'Sim' if fold_refit.get('enabled') else 'Não'}",
            f"   Features OWL candidatas: {len(semantic.get('candidate_semantic_names') or [])}",
            f"   Melhor subconjunto OOF: {len(candidate_names)}"
            + (f" [{preview}]" if preview else ""),
            f"   OOF MLP base — Acc: {_pct(baseline.get('accuracy'))} | "
            f"BA: {_pct(baseline.get('balanced_accuracy'))} | "
            f"Macro-F1: {_pct(baseline.get('macro_f1'))}",
            f"   OOF MLP + OWL — Acc: {_pct(candidate.get('accuracy'))} | "
            f"BA: {_pct(candidate.get('balanced_accuracy'))} | "
            f"Macro-F1: {_pct(candidate.get('macro_f1'))}",
            f"   Utilidade base: {_num(baseline.get('utility'))} | "
            f"utilidade OWL: {_num(candidate.get('utility'))} | "
            f"ganho líquido: {_num(semantic.get('utility_gain'))}",
        ])
    issues = state.get("quality_issues") or []
    if issues:
        lines.append("   Problemas da OWL: " + " | ".join(map(str, issues)))
    return "\n".join(lines)


__all__ = ["build_ontology_status_text"]
