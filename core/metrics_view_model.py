"""Modelo de dados da UI de métricas, independente de PyQt."""
from __future__ import annotations

from typing import Any, Dict, List

from core.algorithm_identity import identity_for


def build_metrics_rows(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    precision = results.get("precision") or {}
    fidelity = results.get("fidelity") or {}
    identities = {
        "trepan_original": ("trepan_original", "TREPAN Original", "MLP Original", "original"),
        "c45_j48": ("c45_j48", "C4.5-Nativo", "MLP Original", "original"),
        "trepan_reloaded": ("trepan_reloaded", "TREPAN Reloaded", None, "enriched"),
    }
    rows = []
    for result_key, (identity_key, label, default_oracle, default_space) in identities.items():
        block = precision.get(result_key)
        if not block:
            continue
        fid = fidelity.get(result_key) or {}
        ref = fid.get("fidelity_reference") or default_oracle
        identity = identity_for(identity_key)
        rows.append({
            "key": result_key,
            "model": label,
            "implementation": identity.implementation,
            "canonical": identity.canonical,
            "oracle": ref or "não aplicável",
            "feature_space": fid.get("feature_space", default_space),
            "accuracy": float(block.get("accuracy", 0.0)),
            "balanced_accuracy": float(block.get("balanced_accuracy", 0.0)),
            "macro_f1": float(block.get("f1_macro", 0.0)),
            "precision_macro": float(block.get("precision_macro", block.get("precision", 0.0))),
            "precision_weighted": float(block.get("precision", 0.0)),
            "fidelity": (
                float(fid["overall_fidelity"])
                if fid.get("overall_fidelity") is not None else None
            ),
        })
    return rows


def claim_banner(results: Dict[str, Any]) -> Dict[str, str]:
    validation = results.get("scientific_validation") or {}
    supported = validation.get("claim_guard") == "superiority_supported"
    if supported:
        return {
            "status": "supported",
            "text": "Superioridade suportada neste teste bloqueado; confirme em validação externa.",
        }
    return {
        "status": "not_supported",
        "text": "Superioridade não demonstrada. Os valores são resultados desta execução, não prova clínica.",
    }
