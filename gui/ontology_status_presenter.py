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


def _num(value: Any, decimals: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{decimals}f}".replace(".", ",")
    except (TypeError, ValueError):
        return "N/A"


def build_semantic_diagnostics_text(
    quality_report: Optional[Dict[str, Any]],
    enrichment_report: Optional[Dict[str, Any]],
) -> str:
    """Painel completo: ontologia, features semânticas, validação do MLP e TREPAN.

    Nunca mostra apenas o veredicto: cada decisão vem com a causa e os números.
    """
    quality = dict(quality_report or {})
    rep = dict(enrichment_report or {})
    metrics = dict(quality.get("metrics") or {})
    stages = dict(rep.get("stages") or {})
    state = build_ontology_stage_status(quality_report, {}, rep)
    reasoner = dict(rep.get("reasoner") or quality.get("reasoner") or {})
    tbox = ((metrics.get("knowledge_split") or {}).get("tbox")) or {}
    richness = metrics.get("semantic_richness") or {}

    if reasoner.get("reasoner_used") or reasoner.get("executed"):
        verdict = "CONSISTENTE" if reasoner.get("consistent") else "INCONSISTENTE"
        reasoner_line = (
            f"{verdict} ({_num(reasoner.get('duration_seconds'), 2)} s, "
            f"{reasoner.get('inferred_axioms_count', 0)} axiomas inferidos)"
        )
    else:
        reasoner_line = "NÃO EXECUTADO — fallback: " + str(reasoner.get("fallback") or "axiomas explícitos")

    lines = [
        "ONTOLOGIA",
        f"   Estado estrutural: {state['ontology_structural_status']}",
        f"   Reasoner: {reasoner_line}",
        f"   ARFF ↔ OWL: {state['mapped_features']} / {state['total_features']} "
        f"({_pct(state['feature_coverage'])}) — estado do mapeamento: {state['mapping_status']}",
        f"   Ambíguos: {metrics.get('ambiguous_matches', 'N/A')} | colisões: {metrics.get('entity_collisions', 'N/A')} "
        f"| razão genérica: {_pct(metrics.get('generic_match_ratio'))}",
        f"   TBox: {tbox.get('classes', 'N/A')} classes, {tbox.get('datatype_properties', 'N/A')} propriedades de dados, "
        f"{tbox.get('object_properties', 'N/A')} de objeto",
        f"   ABox: {state['abox_status']}",
        f"   Riqueza semântica: {richness.get('level', 'N/A')}"
        + (f" ({', '.join(richness.get('knowledge_sources') or []) or 'só taxonomia'})" if richness else ""),
    ]
    novelty = stages.get("B_novelty") or {}
    screening = stages.get("C_screening") or {}
    if novelty:
        removed = {k[8:]: v for k, v in novelty.items() if k.startswith("removed_") and v}
        lines += [
            "",
            "FEATURES SEMÂNTICAS",
            f"   Geradas: {novelty.get('generated', 'N/A')}",
            "   Removidas: " + (", ".join(f"{k}={v}" for k, v in removed.items()) or "nenhuma"),
            f"   Retidas (novidade): {novelty.get('retained', 'N/A')} "
            f"(OWL: {novelty.get('retained_ontology_knowledge', 'N/A')}, "
            f"estatísticas: {novelty.get('retained_statistical', 'N/A')})",
            f"   Estáveis: {len(screening.get('stable_features') or [])}",
            f"   Selecionadas: {len(rep.get('selected_semantic_features') or [])}",
        ]
    cmp = stages.get("D_mlp_comparison")
    if cmp:
        base, onto, delta = cmp["base"], cmp["with_owl"], cmp["delta"]
        ci = cmp.get("utility_gain_ci") or [None, None]
        lines += ["", "VALIDAÇÃO SEMÂNTICA DO MLP", f"   {'':24}{'Base':>10}{'+ OWL':>10}{'Δ':>10}"]
        for label, key in (("Accuracy", "accuracy"), ("Balanced Accuracy", "balanced_accuracy"),
                           ("Macro-F1", "macro_f1"), ("Recall macro", "recall_macro"),
                           ("Recall minoritária", "minority_recall"), ("Precisão macro", "precision_macro")):
            lines.append(f"   {label:24}{_num(base[key], 3):>10}{_num(onto[key], 3):>10}{_num(delta[key], 3):>10}")
        lines += [
            f"   {'Utilidade':24}{_num(base['utility'], 3):>10}{_num(onto['utility'], 3):>10}"
            f"{_num(cmp['utility_gain'], 3):>10}",
            f"   Ganho líquido: {_num(cmp['utility_gain'], 4)} "
            f"(IC {_pct(cmp.get('confidence'))}: [{_num(ci[0], 4)}; {_num(ci[1], 4)}])",
        ]
    if rep:
        lines += ["", f"   Decisão: {rep.get('decision')}", f"   Causa: {rep.get('decision_reason')}"]
    lines += [
        "",
        "SEMÂNTICA PARA O TREPAN",
        f"   Disponível: {'SIM' if state['semantic_trepan_available'] else 'NÃO'} "
        f"({state['semantic_trepan_reason']})",
        f"   Relações inferidas pelo reasoner: {reasoner.get('inferred_axioms_count', 'N/A')}",
        f"   Conhecimento ontológico explorável: "
        f"{'SIM' if rep.get('ontology_knowledge_available_for_trepan') else 'NÃO / N/A'}",
        f"   MLP com OWL aceite: {'SIM' if state['semantic_mlp_accepted'] else 'NÃO'} "
        "(independente da disponibilidade para o TREPAN)",
    ]
    return "\n".join(lines)


def build_ontology_status_text(
    quality_report: Optional[Dict[str, Any]],
    acceptance: Optional[Dict[str, Any]],
    *,
    selected_oracle_label: str = "MLP Original",
    enrichment_report: Optional[Dict[str, Any]] = None,
) -> str:
    acc = dict(acceptance or {})
    state = build_ontology_stage_status(quality_report, acc, enrichment_report)
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
    if enrichment_report:
        lines.append("")
        lines.append(build_semantic_diagnostics_text(quality_report, enrichment_report))
    return "\n".join(lines)


__all__ = ["build_ontology_status_text", "build_semantic_diagnostics_text"]
