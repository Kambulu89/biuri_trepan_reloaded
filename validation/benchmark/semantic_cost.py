"""Métricas de custo e de ganho do enriquecimento ontológico (contrato EQ v2.0, S0–S3). Funções puras, sem estado, sem dependência de datasets.

Os ganhos calculam-se SEMPRE em conjuntos de avaliação passados explicitamente (validação interna na otimização; teste externo só no relato final) e
nunca alimentam a seleção quando o conjunto é o teste. Um ganho observado numa amostra é evidência observada, não inferência (``evidence_level``).
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np

GAIN_STATUSES = ("SEMANTIC_GAIN_OBSERVED", "SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN", "SEMANTIC_HARM_OBSERVED", "NO_SEMANTIC_ATTEMPT")


def tree_complexity(tree) -> Dict[str, int]:
    nodes = list(tree.iter_nodes())
    leaves = [n for n in nodes if getattr(n, "is_leaf", False)]
    return {"nodes": len(nodes), "leaves": len(leaves), "rules": len(leaves)}


def semantic_gain_report(tree_owl, tree_ref, X_eval, y_eval, oracle_pred_eval, *, tol: float = 0.0) -> Dict[str, Any]:
    """Compara a árvore com ontologia (``tree_owl``) com o Original equivalente (``tree_ref``) num conjunto de avaliação.

    ``oracle_pred_eval`` = previsões do MLP congelado em ``X_eval`` (fidelidade); ``y_eval`` = rótulos reais (accuracy).
    Ganho estrito: > ``tol`` (por omissão 0, δ=0). Empates NÃO são ganho.
    """
    X_eval = np.asarray(X_eval, dtype=float)
    p_owl, p_ref = np.asarray(tree_owl.predict(X_eval)), np.asarray(tree_ref.predict(X_eval))
    y_eval, orc = np.asarray(y_eval), np.asarray(oracle_pred_eval)
    acc_owl, acc_ref = float(np.mean(p_owl == y_eval)), float(np.mean(p_ref == y_eval))
    fid_owl, fid_ref = float(np.mean(p_owl == orc)), float(np.mean(p_ref == orc))
    g_acc, g_fid = acc_owl - acc_ref, fid_owl - fid_ref
    c_owl, c_ref = tree_complexity(tree_owl), tree_complexity(tree_ref)
    attempted = bool(getattr(tree_owl, "oracle_accounting_", {}).get("semantic_attempt_executed", False))
    if not attempted:
        status = "NO_SEMANTIC_ATTEMPT"
    elif (g_acc > tol or g_fid > tol) and min(g_acc, g_fid) >= -tol:
        status = "SEMANTIC_GAIN_OBSERVED"
    elif g_acc < -tol or g_fid < -tol:
        status = "SEMANTIC_HARM_OBSERVED" if not (g_acc > tol or g_fid > tol) else "SEMANTIC_GAIN_OBSERVED"
    else:
        status = "SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN"
    return {
        "semantic_gain_accuracy": g_acc, "semantic_gain_fidelity": g_fid,
        "accuracy_owl": acc_owl, "accuracy_ref": acc_ref, "fidelity_owl": fid_owl, "fidelity_ref": fid_ref,
        "fidelity_non_inferior": bool(fid_owl >= fid_ref), "fidelity_strictly_superior": bool(fid_owl > fid_ref),
        "accuracy_strictly_superior": bool(acc_owl > acc_ref),
        "complexity_owl": c_owl, "complexity_ref": c_ref, "complexity_diff_nodes": c_owl["nodes"] - c_ref["nodes"],
        "predictions_identical": bool(np.array_equal(p_owl, p_ref)),
        "semantic_attempt_status": status, "evidence_level": "observed_only",
    }


def cost_report(tree_owl, ref_total_oracle_samples: Optional[int] = None, ref_total_oracle_calls: Optional[int] = None) -> Dict[str, Any]:
    """Campos de custo do contrato (a partir de ``oracle_accounting_``) + custo adicional face ao Original de referência."""
    acc = dict(getattr(tree_owl, "oracle_accounting_", {}) or {})
    keys = ("original_oracle_calls", "original_oracle_samples", "semantic_oracle_calls", "semantic_oracle_samples", "reused_oracle_calls",
            "total_oracle_calls", "total_oracle_samples", "semantic_attempt_executed", "semantic_effect_observed")
    out = {k: acc.get(k) for k in keys}
    out["reused_oracle_samples"] = acc.get("reused_oracle_samples")
    out["extra_oracle_samples_vs_original"] = (int(acc["total_oracle_samples"]) - int(ref_total_oracle_samples)) \
        if ref_total_oracle_samples is not None and "total_oracle_samples" in acc else None
    out["extra_oracle_calls_vs_original"] = (int(acc["total_oracle_calls"]) - int(ref_total_oracle_calls)) \
        if ref_total_oracle_calls is not None and "total_oracle_calls" in acc else None
    return out


def efficiency_report(gain: Dict[str, Any], cost: Dict[str, Any], tree_owl=None) -> Dict[str, Any]:
    """Eficiência semântica: ganho por 1 000 amostras adicionais ao oráculo; taxa de aceitação de candidatos semânticos."""
    extra = cost.get("extra_oracle_samples_vs_original")
    per_k = (lambda g: (g / (extra / 1000.0)) if extra and extra > 0 else None)
    out = {"fidelity_gain_per_1k_extra_samples": per_k(gain["semantic_gain_fidelity"]),
           "accuracy_gain_per_1k_extra_samples": per_k(gain["semantic_gain_accuracy"]),
           "complexity_diff_nodes": gain["complexity_diff_nodes"]}
    if tree_owl is not None and hasattr(tree_owl, "semantic_audit_summary"):
        s = tree_owl.semantic_audit_summary()
        ev = max(1, int(s.get("evaluated_splits", 0)))
        att = int(s.get("error_focused_interventions_attempted", 0))
        out["semantic_split_acceptance_rate"] = s.get("semantic_changed_splits", 0) / ev
        out["efsr_acceptance_rate"] = (s.get("error_focused_interventions_accepted", 0) / att) if att else None
    return out


def no_effect_rate(reports: Iterable[Dict[str, Any]]) -> Optional[float]:
    """Proporção de tentativas semânticas executadas sem ganho observado (``SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN``)."""
    att = [r for r in reports if r.get("semantic_attempt_status") != "NO_SEMANTIC_ATTEMPT"]
    return (sum(r["semantic_attempt_status"] == "SEMANTIC_ATTEMPT_NO_OBSERVED_GAIN" for r in att) / len(att)) if att else None
