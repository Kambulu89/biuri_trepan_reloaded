"""Resumo de paragem de uma árvore TREPAN para a interface (só leitura).

Deriva, a partir do estado que o próprio TREPAN já regista (motivos por folha, ``pruning_summary_``,
orçamentos), o dicionário consumido por ``core.experiment_builders.tree_diagnostics``. Nada é
recalculado nem altera a árvore.
"""
from __future__ import annotations

from typing import Any, Dict

from core.tree_build_report import leaf_stop_reasons


def stop_summary_of(model: Any) -> Dict[str, Any]:
    if model is None or getattr(model, "root_", None) is None:
        return {}
    pruning = dict(getattr(model, "pruning_summary_", {}) or {})
    nodes_after = pruning.get("nodes_after_pruning", getattr(model, "node_count_", None))
    return {
        "loop_end_reason": "node_budget_exhausted" if getattr(model, "max_nodes_reached_", False) else "no_expandable_nodes_left",
        "node_budget": getattr(model, "max_nodes", None),
        "nodes_before_pruning": pruning.get("nodes_before_pruning", getattr(model, "nodes_raw_", None)),
        "nodes_after_pruning": nodes_after,
        "query_budget": getattr(model, "max_queries", None),
        "queries_used": getattr(model, "membership_queries_", None),
        "query_budget_exhausted": bool(getattr(model, "query_budget_exhausted_", False)),
        "max_depth": getattr(model, "max_depth", None),
        "stop_reasons": leaf_stop_reasons(model),
    }
