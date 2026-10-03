"""Relatório científico de construção de árvores (C4.5, TREPAN Original, TREPAN Reloaded).

Responde, para cada árvore, às perguntas de auditoria: quantas membership
queries foram usadas, se o orçamento acabou, se a expansão foi best-first,
quantos candidatos simples/m-of-n foram avaliados, porque cada folha parou,
o que a poda removeu e qual a diferença entre *fidelity* (face ao oráculo) e
*accuracy* (face aos rótulos reais). Nada aqui toca em dados de teste para
decidir splits ou podas.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Optional

import numpy as np

STUMP_MAX_NODES = 3


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


# ---------------------------------------------------------------- complexidade
def trepan_complexity(tree) -> Dict[str, Any]:
    nodes = list(tree.iter_nodes())
    leaves = [n for n in nodes if n.is_leaf]
    internal = [n for n in nodes if not n.is_leaf]
    mofn = [n.test for n in internal if len(n.test.literals) > 1 or n.test.m > 1]
    features = sorted({lit.feature for n in internal for lit in n.test.literals})
    rule_lengths = [n.depth for n in leaves]
    return {
        "node_count": len(nodes),
        "leaf_count": len(leaves),
        "internal_node_count": len(internal),
        "depth": int(tree.get_depth()),
        "average_leaf_depth": float(np.mean(rule_lengths)) if rule_lengths else 0.0,
        "average_rule_length": float(np.mean(rule_lengths)) if rule_lengths else 0.0,
        "m_of_n_count": len(mofn),
        "mean_mofn_n": float(np.mean([len(t.literals) for t in mofn])) if mofn else 0.0,
        "mean_mofn_m": float(np.mean([t.m for t in mofn])) if mofn else 0.0,
        "features_used": features,
        "n_features_used": len(features),
    }


def c45_complexity(tree) -> Dict[str, Any]:
    rules = tree.iter_rules()
    features = sorted({int(c[0]) for r in rules for c in r["conditions"]})
    leaves = len(rules)
    return {
        "node_count": int(_c45_count(tree.root_)),
        "leaf_count": leaves,
        "internal_node_count": int(_c45_count(tree.root_) - leaves),
        "depth": int(tree.get_depth()),
        "average_leaf_depth": float(np.mean([len(r["conditions"]) for r in rules])) if rules else 0.0,
        "average_rule_length": float(np.mean([len(r["conditions"]) for r in rules])) if rules else 0.0,
        "m_of_n_count": 0,
        "mean_mofn_n": 0.0,
        "mean_mofn_m": 0.0,
        "features_used": features,
        "n_features_used": len(features),
    }


def _c45_count(node) -> int:
    return 1 if node.is_leaf else 1 + sum(_c45_count(c) for c in node.children.values())


def _c45_leaf_stops(node, acc: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    acc = {} if acc is None else acc
    if node.is_leaf:
        key = node.stop_reason or "STOP_NO_VALID_SPLIT"
        acc[key] = acc.get(key, 0) + 1
    else:
        for child in node.children.values():
            _c45_leaf_stops(child, acc)
    return acc


# --------------------------------------------------------------- stump / paragem
def leaf_stop_reasons(tree) -> Dict[str, int]:
    """Histograma de motivos de paragem das folhas finais."""
    out: Dict[str, int] = {}
    for node in tree.iter_nodes():
        if node.is_leaf:
            key = node.stop_reason or "STOP_UNRECORDED"
            out[key] = out.get(key, 0) + 1
    return out


def stump_diagnostic(tree) -> Optional[Dict[str, Any]]:
    """Diagnóstico ``TREE_STUMP_DIAGNOSTIC`` para árvores com <= 3 nós.

    Um stump não é automaticamente um erro: o diagnóstico só explica porquê.
    """
    nodes = list(tree.iter_nodes())
    if len(nodes) > STUMP_MAX_NODES:
        return None
    leaves = [n for n in nodes if n.is_leaf]
    reasons = {str(n.node_id): {"stop_reason": n.stop_reason, "detail": _jsonable(n.stop_detail)} for n in leaves}
    codes = {n.stop_reason for n in leaves}
    pruned_children = any(n.stop_reason == "STOP_PRUNED" for n in leaves)
    return {
        "alert": "TREE_STUMP_DIAGNOSTIC",
        "node_count": len(nodes),
        "depth": int(tree.get_depth()),
        "query_budget_exhausted": bool(getattr(tree, "query_budget_exhausted_", False)) or "STOP_QUERY_BUDGET_EXHAUSTED" in codes,
        "max_depth_reached": "STOP_MAX_DEPTH" in codes,
        "max_nodes_reached": "STOP_MAX_NODES" in codes,
        "min_samples": "STOP_MIN_SAMPLES" in codes,
        "no_valid_split": "STOP_NO_VALID_SPLIT" in codes,
        "min_gain": "STOP_MIN_GAIN" in codes,
        "pure_children": "STOP_PURE_NODE" in codes,
        "pruning_collapsed_tree": bool(pruned_children or getattr(tree, "nodes_raw_", len(nodes)) > len(nodes)),
        "error": bool(codes & {"STOP_NUMERICAL_FAILURE", "STOP_QUERY_GENERATION_FAILURE"}),
        "leaf_stop_reasons": reasons,
        "legitimate_by_construction": not bool(
            codes & {"STOP_QUERY_BUDGET_EXHAUSTED", "STOP_NUMERICAL_FAILURE", "STOP_QUERY_GENERATION_FAILURE"}
        ),
    }


# ------------------------------------------------------------------- relatório
def trepan_build_report(
    tree,
    *,
    algorithm: str,
    oracle_name: str,
    oracle_type: Optional[str] = None,
    X_eval=None,
    y_oracle_eval=None,
    y_real_eval=None,
    oracle_accuracy: Optional[float] = None,
    dataset_hash: Optional[str] = None,
    ontology_hash: Optional[str] = None,
) -> Dict[str, Any]:
    """Relatório completo de uma árvore TREPAN (Original ou Reloaded).

    ``fidelity`` é sempre face ao oráculo (``y_oracle_eval``); ``accuracy_real``
    é face a ``y_real_eval``. As duas métricas nunca se misturam.
    """
    totals = dict(getattr(tree, "candidate_totals_", {}) or {})
    splits = list(getattr(tree, "split_audit_", []) or [])
    simple_wins = sum(1 for s in splits if s.get("n", 1) <= 1 and s.get("m", 1) <= 1)
    mofn_wins = len(splits) - simple_wins
    complexity = trepan_complexity(tree)
    expansion = list(getattr(tree, "expansion_log_", []) or [])
    best_first_ok = all(item["is_best_first_choice"] for item in expansion) if expansion else True
    budget = int(tree.max_queries)
    used = int(tree.membership_queries_)
    report: Dict[str, Any] = {
        "ALGORITHM": algorithm,
        "oracle": {
            "oracle_name": oracle_name,
            "oracle_type": oracle_type or getattr(tree, "oracle_info_", {}).get("oracle_type"),
            "oracle_accuracy": oracle_accuracy,
            "oracle_feature_space": int(tree.n_features_in_),
            "uses_real_labels_for_training": False,
        },
        "STRUCTURE": {
            "nodes_raw": int(getattr(tree, "nodes_raw_", complexity["node_count"])),
            "nodes_final": complexity["node_count"],
            "leaves": complexity["leaf_count"],
            "depth": complexity["depth"],
            "nodes_created": int(getattr(tree, "nodes_created_", 0)),
            "nodes_expanded": int(getattr(tree, "nodes_expanded_", 0)),
            "configured_max_depth": tree.max_depth,
            "achieved_depth": complexity["depth"],
            "max_nodes": int(tree.max_nodes),
        },
        "CONSTRUCTION": {
            "query_budget_initial": budget,
            "query_budget_used": used,
            "query_budget_remaining": max(0, budget - used),
            "query_budget_exhausted": bool(getattr(tree, "query_budget_exhausted_", False)),
            "budget_starved_nodes": list(getattr(tree, "budget_starved_nodes_", [])),
            "effective_min_sample": int(tree.effective_min_sample_),
            "best_first_verified": bool(best_first_ok),
            "simple_candidates_generated": int(totals.get("simple_generated", 0)),
            "simple_candidates_evaluated": int(totals.get("simple_evaluated", 0)),
            "m_of_n_candidates_generated": int(totals.get("mofn_generated", 0)),
            "m_of_n_candidates_evaluated": int(totals.get("mofn_evaluated", 0)),
            "semantic_candidates": int(totals.get("semantic_candidates", 0)),
            "simple_split_wins": int(simple_wins),
            "m_of_n_split_wins": int(mofn_wins),
            "m_of_n_final_tree_count": complexity["m_of_n_count"],
            "min_gain": float(getattr(tree, "min_gain", 0.0)),
        },
        "PRUNING": dict(getattr(tree, "pruning_summary_", {}) or {}),
        "pruning_audit": _jsonable(getattr(tree, "pruning_audit_", [])),
        "STOP_REASONS": leaf_stop_reasons(tree),
        "COMPLEXITY": complexity,
        "TIME": {
            "training_time": float(getattr(tree, "training_time_", 0.0)),
            "query_time": float(getattr(tree, "query_time_", 0.0)),
            "split_search_time": float(getattr(tree, "split_search_time_", 0.0)),
            "m_of_n_search_time": float(getattr(tree, "m_of_n_search_time_", 0.0)),
            "pruning_time": float(getattr(tree, "pruning_time_", 0.0)),
        },
        "REPRODUCIBILITY": {
            "random_seed": tree.random_state,
            "dataset_hash": dataset_hash,
            "ontology_hash": ontology_hash,
            "parameters": {k: _jsonable(v) for k, v in tree.get_params().items()
                           if isinstance(v, (int, float, str, bool, type(None)))},
        },
        "expansion_log": _jsonable(expansion),
        "node_log": _node_log(tree),
        "STUMP": stump_diagnostic(tree),
    }
    if X_eval is not None:
        pred = np.asarray(tree.predict(X_eval))
        metrics: Dict[str, Any] = {}
        if y_oracle_eval is not None:
            metrics["fidelity"] = float(np.mean(pred == np.asarray(y_oracle_eval)))
            metrics["fidelity_target"] = oracle_name
        if y_real_eval is not None:
            metrics["accuracy_real"] = float(np.mean(pred == np.asarray(y_real_eval)))
        report["METRICS"] = metrics
    return _jsonable(report)


def _node_log(tree) -> list:
    rows = []
    for node in tree.iter_nodes():
        row = {
            "node_id": node.node_id, "depth": node.depth, "is_leaf": node.is_leaf,
            "prediction": node.prediction, "reach": float(node.reach),
            "stop_reason": node.stop_reason, "stop_detail": node.stop_detail,
        }
        row.update(node.stats)
        if not node.is_leaf:
            row["test"] = node.test.text(getattr(tree, "feature_names_in_", None))
        rows.append(row)
    return rows


def c45_build_report(
    tree,
    *,
    X_eval=None,
    y_real_eval=None,
    oracle_eval=None,
    oracle_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Relatório do C4.5: treinado exclusivamente com rótulos reais."""
    complexity = c45_complexity(tree)
    raw = getattr(tree, "root_raw_", None)
    raw_count = _c45_count(raw) if raw is not None else complexity["node_count"]
    report: Dict[str, Any] = {
        "ALGORITHM": "C4.5",
        "implementation": type(tree).__name__,
        "split_criterion": "gain_ratio",
        "training_target": "real labels (y_train)",
        "uses_oracle": False,
        "uses_membership_queries": False,
        "STRUCTURE": {
            "nodes_raw": int(raw_count), "nodes_final": complexity["node_count"],
            "leaves": complexity["leaf_count"], "depth": complexity["depth"],
        },
        "PRUNING": {
            "nodes_before_pruning": int(raw_count),
            "nodes_after_pruning": complexity["node_count"],
            "pruned_nodes": int(raw_count - complexity["node_count"]),
            "pruned_subtrees": len(getattr(tree, "pruning_audit_", [])),
            "uses_test_data": False,
        },
        "pruning_audit": _jsonable(getattr(tree, "pruning_audit_", [])),
        "STOP_REASONS": _c45_leaf_stops(tree.root_),
        "COMPLEXITY": complexity,
    }
    if X_eval is not None:
        pred = np.asarray(tree.predict(X_eval))
        metrics: Dict[str, Any] = {}
        if y_real_eval is not None:
            metrics["accuracy_real"] = float(np.mean(pred == np.asarray(y_real_eval)))
        if oracle_eval is not None:
            # Informativo: concordância com o MLP. Não é a métrica do C4.5.
            metrics["agreement_with_oracle_informative"] = float(np.mean(pred == np.asarray(oracle_eval)))
            metrics["agreement_target"] = oracle_name
        report["METRICS"] = metrics
    return _jsonable(report)


def format_report(report: Dict[str, Any]) -> str:
    lines = [f"ALGORITHM: {report['ALGORITHM']}"]
    if "oracle" in report:
        o = report["oracle"]
        lines.append(f"ORACLE: {o['oracle_name']} ({o.get('oracle_type')})")
    s = report["STRUCTURE"]
    lines.append(f"STRUCTURE: nodes_raw={s['nodes_raw']} nodes_final={s['nodes_final']} "
                 f"leaves={s['leaves']} depth={s['depth']}")
    if "CONSTRUCTION" in report:
        c = report["CONSTRUCTION"]
        lines.append(f"CONSTRUCTION: queries={c['query_budget_used']}/{c['query_budget_initial']} "
                     f"budget_exhausted={c['query_budget_exhausted']} "
                     f"simple_evaluated={c['simple_candidates_evaluated']} "
                     f"m_of_n_evaluated={c['m_of_n_candidates_evaluated']} "
                     f"m_of_n_selected={c['m_of_n_split_wins']} best_first={c['best_first_verified']}")
    p = report.get("PRUNING", {})
    if p:
        lines.append(f"PRUNING: before={p.get('nodes_before_pruning')} after={p.get('nodes_after_pruning')}")
    lines.append("STOP REASONS: " + ", ".join(f"{k}={v}" for k, v in sorted(report["STOP_REASONS"].items())))
    if report.get("METRICS"):
        lines.append("METRICS: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in report["METRICS"].items()))
    if report.get("STUMP"):
        st = report["STUMP"]
        lines.append(f"TREE_STUMP_DIAGNOSTIC: nodes={st['node_count']} "
                     f"legitimate_by_construction={st['legitimate_by_construction']} "
                     f"budget_exhausted={st['query_budget_exhausted']}")
        for nid, info in st["leaf_stop_reasons"].items():
            lines.append(f"  leaf {nid}: {info['stop_reason']} {info['detail']}")
    if "TIME" in report:
        lines.append("TIME: " + ", ".join(f"{k}={v:.3f}s" for k, v in report["TIME"].items()))
    return "\n".join(lines)


# ------------------------------------------------------------------ cache / hash
def tree_cache_key(
    *,
    algorithm: str,
    dataset_hash: str,
    oracle_hash: str,
    ontology_hash: str = "none",
    parameters: Optional[Dict[str, Any]] = None,
    code_version: str = "v9.3",
) -> str:
    """Chave de cache de árvore: qualquer parâmetro relevante muda a chave (CACHE MISS).

    ``parameters`` deve conter query_budget, max_depth, max_nodes, min_samples,
    min_gain, semantic_weight, pruning_config e seed.
    """
    payload = {
        "algorithm": algorithm, "dataset": dataset_hash, "oracle": oracle_hash,
        "ontology": ontology_hash, "parameters": _jsonable(parameters or {}),
        "code_version": code_version,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()
