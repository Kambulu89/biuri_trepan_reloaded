"""Estado consolidado das etapas ontológicas do BIURI/TREPAN Reloaded.

A validade estrutural da OWL, a utilidade das features derivadas para o MLP e a
aceitação de um professor ontológico são decisões independentes. Este módulo
mantém essa separação sem depender de nomes ou características de datasets.
"""
from __future__ import annotations

from typing import Any, Dict, Optional


def build_ontology_stage_status(
    quality_report: Optional[Dict[str, Any]],
    acceptance: Optional[Dict[str, Any]],
    enrichment_report: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    quality = dict(quality_report or {})
    acc = dict(acceptance or {})
    metrics = dict(quality.get("metrics") or {})

    total = int(metrics.get("total_features") or 0)
    mapped = int(metrics.get("mapped_features") or 0)
    coverage = metrics.get("feature_coverage")
    if coverage is None:
        coverage = mapped / max(total, 1) if total else 0.0

    quality_accepted = bool(quality.get("accepted"))
    feature_gate_accepted = bool(acc.get("ontology_feature_gate_accepted"))
    teacher_has_ontology = bool(acc.get("teacher_has_ontology"))
    oracle_gate_accepted = bool(acc.get("oracle_gate_accepted"))
    teacher_accepted = bool(
        acc.get("accepted") and teacher_has_ontology and oracle_gate_accepted
    )

    # ---- Estados independentes (A..E): validade != utilidade -----------------
    abox = dict(quality.get("abox") or {})
    abox_status = abox.get("status") or ("SAFE" if quality_accepted else "NOT_EVALUATED")
    ambiguous = int(metrics.get("ambiguous_matches") or 0)
    if not quality:
        mapping_status = "NOT_EVALUATED"
    elif mapped and ambiguous == 0:
        mapping_status = "VALID"
    elif mapped:
        mapping_status = "VALID_WITH_AMBIGUITY"
    else:
        mapping_status = "NO_VALID_MATCHES"
    enrich = dict(enrichment_report or {})
    novelty = (enrich.get("stages") or {}).get("B_novelty") or {}
    if novelty:
        semantic_novelty = novelty.get("status", "NOT_EVALUATED")
    elif acc.get("semantic_utility_status") == "REJECT_NO_INFORMATIONAL_NOVELTY":
        semantic_novelty = "NO_NOVEL_FEATURES"
    elif acc.get("semantic_utility_status"):
        semantic_novelty = "VALID"
    else:
        semantic_novelty = "NOT_EVALUATED"
    mlp_status = enrich.get("decision") or acc.get("semantic_utility_status") or "NOT_EVALUATED"
    semantic_mlp_accepted = bool(
        enrich.get("semantic_mlp_accepted") if enrich else feature_gate_accepted
    )
    # O TREPAN pode usar a ontologia (grupos, relatedness, profundidade) desde que ela seja
    # estruturalmente válida, bem mapeada e sem risco de leakage, mesmo que o MLP não ganhe.
    abox_safe = abox.get("accepted", True) is not False
    trepan_available = bool(quality_accepted and abox_safe and mapped > 0)
    if trepan_available:
        trepan_reason = "structure_valid_mapped_and_leak_free"
    elif not quality_accepted:
        trepan_reason = f"ontology_quality:{quality.get('status') or 'NOT_EVALUATED'}"
    elif not abox_safe:
        trepan_reason = f"abox:{abox_status}"
    else:
        trepan_reason = "no_mapped_features"

    return {
        "ontology_structural_status": "VALID" if quality_accepted else (quality.get("status") or "NOT_EVALUATED"),
        "abox_status": abox_status,
        "mapping_status": mapping_status,
        "semantic_novelty_status": semantic_novelty,
        "mlp_enrichment_status": mlp_status,
        "semantic_mlp_accepted": semantic_mlp_accepted,
        "semantic_trepan_available": trepan_available,
        "semantic_trepan_reason": trepan_reason,
        "ontology_quality_accepted": quality_accepted,
        "ontology_quality_status": quality.get("status") or "NOT_EVALUATED",
        "ontology_structural_available": quality_accepted,
        "trepan_semantic_use_allowed": quality_accepted,
        "mapped_features": mapped,
        "total_features": total,
        "feature_coverage": float(coverage),
        "ontology_feature_engineering_accepted": feature_gate_accepted,
        "semantic_utility_status": acc.get("semantic_utility_status") or "NOT_EVALUATED",
        "ontological_teacher_accepted": teacher_accepted,
        "oracle_gate_accepted": oracle_gate_accepted,
        "teacher_has_ontology": teacher_has_ontology,
        "quality_issues": list(quality.get("issues") or []),
        "quality_warnings": list(quality.get("warnings") or []),
    }


__all__ = ["build_ontology_stage_status"]
