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

    return {
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
