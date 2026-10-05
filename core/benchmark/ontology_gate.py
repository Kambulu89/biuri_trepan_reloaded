"""Quality gate do uso REAL da ontologia por árvore: carregar uma OWL não é prova de efeito semântico.

``ontology_effectively_used`` só é verdadeiro com evidência objetiva de intervenção semântica na árvore final (splits cuja
decisão foi influenciada pela ontologia, features semânticas usadas pela árvore ou regras que as contêm) e SEM espelhamento.
Genérico (não conhece nenhum dataset).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def ontology_gate_fields(*, arm_uses_ontology: bool, control: bool, ctx, structure: Dict[str, Any], tree,
                         final_audit_rows, semantic_split_count: int, mirror_applied: bool) -> Dict[str, Any]:
    """Campos do quality gate para UMA árvore. ``arm_uses_ontology=False`` (ex.: Reloaded Core) => tudo zero/False."""
    info = dict(getattr(ctx, "info", {}) or {})
    n_orig = len(getattr(ctx, "orig_idx", []) or [])
    onto_idx = set(getattr(ctx, "onto_idx", []) or [])
    mapped = int(info.get("mapped_feature_count", 0) or 0)
    base = dict(
        ontology_loaded=bool(getattr(ctx, "available", False) or getattr(ctx, "ontology_valid", None) is not None),
        ontology_valid=bool(getattr(ctx, "ontology_valid", None)),
        ontology_mapped=bool(mapped > 0 or onto_idx),
        mapped_feature_count=mapped, unmapped_feature_count=max(0, n_orig - mapped),
        mapping_rate=float(mapped / n_orig) if n_orig else 0.0,
        semantic_features_generated=int(len(onto_idx)),
        reasoning_applied=bool(info.get("reasoning_applied", False) or float(getattr(ctx, "reasoner_time", 0.0) or 0.0) > 0.0),
        mirror_applied=bool(mirror_applied),
    )
    if not arm_uses_ontology:
        # A informação ontológica NÃO entra neste braço: nada é usado, nada é aplicado.
        base.update(ontology_provided_to_arm=False, semantic_features_available=0, semantic_features_used=0,
                    reasoning_applied=False, ontology_influenced_splits=0, semantic_split_count=0, semantic_rule_count=0,
                    ontology_usage_rate=0.0, semantic_decision_impact=0.0, ontology_effectively_used=False,
                    ontology_control_arm=False, semantic_intervention_evidence=False)
        return base
    summary = tree.semantic_audit_summary() if hasattr(tree, "semantic_audit_summary") else {}
    used_onto = [j for j in structure["features_used_idx"] if j in onto_idx]
    influenced = int(summary.get("ontology_influenced_splits", 0)) if not mirror_applied else 0
    # regras (folhas) cujo caminho contém uma feature semântica ou um split influenciado pela ontologia
    influenced_nodes = {r.get("node_id") for r in (final_audit_rows or []) if r.get("ontology_influenced")}
    rules = 0
    if hasattr(tree, "root_"):
        def walk(node, has_sem):
            nonlocal rules
            if node.is_leaf:
                rules += int(has_sem)
                return
            sem = has_sem or node.node_id in influenced_nodes or any(int(l.feature) in onto_idx for l in node.test.literals)
            walk(node.true_child, sem)
            walk(node.false_child, sem)
        walk(tree.root_, False)
    evidence = bool(semantic_split_count > 0 or influenced > 0 or used_onto or rules > 0)
    base.update(
        ontology_provided_to_arm=True, semantic_features_available=int(len(onto_idx)), semantic_features_used=int(len(used_onto)),
        ontology_influenced_splits=influenced, semantic_split_count=int(semantic_split_count), semantic_rule_count=int(rules),
        ontology_usage_rate=float(summary.get("ontology_usage_rate", 0.0)) if not mirror_applied else 0.0,
        semantic_decision_impact=float(summary.get("semantic_decision_impact", 0.0)) if not mirror_applied else 0.0,
        ontology_control_arm=bool(control), semantic_intervention_evidence=bool(evidence and not mirror_applied),
        # só é evidência de utilização real da ontologia se houver intervenção semântica objetiva, sem espelhamento e
        # com ontologia válida; o controlo negativo (OWL permutada) calcula-se igual mas NUNCA é evidência.
        ontology_effectively_used=bool(evidence and not mirror_applied and base["ontology_valid"] and not control),
    )
    return base


ONTOLOGY_CATEGORIES = ("ontology_effectively_used", "ontology_available_but_not_effective", "ontology_invalid_or_unmapped")


def ontology_category(gate: Dict[str, Any]) -> str:
    """Classifica a execução de um braço com ontologia. Só ``ontology_effectively_used`` é evidência (positiva ou negativa)
    sobre o efeito da ontologia; as outras duas significam «sem evidência de utilização semântica», não «não melhora»."""
    if bool(gate.get("ontology_effectively_used")):
        return "ontology_effectively_used"
    unmapped = (not bool(gate.get("ontology_valid"))) or (int(gate.get("mapped_feature_count", 0) or 0) == 0
                                                          and int(gate.get("semantic_features_generated", 0) or 0) == 0)
    return "ontology_invalid_or_unmapped" if unmapped else "ontology_available_but_not_effective"
