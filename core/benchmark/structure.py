"""Estrutura observável de uma árvore (features, raiz, splits, regras) para complexidade e estabilidade entre seeds.

Genérico: funciona para TREPAN (testes m-of-n) e C4.5 (condições simples). Não conhece nenhum dataset. Só descreve a árvore
final; nada aqui decide splits nem toca no teste.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def _bin(feature: int, threshold: float, scale: Optional[np.ndarray]) -> List[Any]:
    """Assinatura (feature, intervalo do limiar em meios desvios-padrão da feature no treino): limiares próximos coincidem."""
    s = float(scale[feature]) if scale is not None and feature < len(scale) and scale[feature] > 0 else 1.0
    return [int(feature), int(np.floor(float(threshold) / (0.5 * s)))]


def trepan_structure(tree, feature_names: Sequence[str], feature_scale: Optional[np.ndarray] = None) -> Dict[str, Any]:
    nodes = list(tree.iter_nodes())
    root = tree.root_
    internal = [n for n in nodes if not n.is_leaf]
    features = sorted({int(l.feature) for n in internal for l in n.test.literals})
    root_features = [] if root.is_leaf else sorted({int(l.feature) for l in root.test.literals})
    splits = sorted({tuple(_bin(l.feature, l.threshold, feature_scale)) for n in internal for l in n.test.literals})
    # nº de literais ao longo do caminho de cada regra (um m-of-n com n literais conta n)
    lengths: List[int] = []

    def walk(node, acc):
        if node.is_leaf:
            lengths.append(acc)
            return
        k = len(node.test.literals)
        walk(node.true_child, acc + k)
        walk(node.false_child, acc + k)

    walk(root, 0)
    names = [str(n) for n in feature_names]
    nm = lambda j: names[j] if j < len(names) else f"x{j}"
    return {
        "features_used_idx": features, "features_used_names": [nm(j) for j in features],
        "root_feature_idx": root_features, "root_feature_names": [nm(j) for j in root_features],
        "split_signatures": [list(s) for s in splits],
        "rule_count": len(lengths),
        "average_rule_literals": float(np.mean(lengths)) if lengths else 0.0,
        "max_rule_literals": int(max(lengths)) if lengths else 0,
        "node_cap_reached": bool(getattr(tree, "max_nodes_reached_", False)),
        "max_nodes": int(getattr(tree, "max_nodes", 0) or 0),
    }


def c45_structure(tree, feature_names: Sequence[str], feature_scale: Optional[np.ndarray] = None) -> Dict[str, Any]:
    rules = tree.iter_rules()
    names = [str(n) for n in feature_names]
    nm = lambda j: names[j] if j < len(names) else f"x{j}"
    feats = sorted({int(c[0]) for r in rules for c in r["conditions"]})
    root = tree.root_
    root_feat = [] if getattr(root, "is_leaf", True) or root.feature is None else [int(root.feature)]
    splits = set()
    for r in rules:
        for c in r["conditions"]:
            thr = next((x for x in c[1:] if isinstance(x, (int, float, np.floating)) and not isinstance(x, bool)), None)
            splits.add(tuple(_bin(int(c[0]), thr, feature_scale)) if thr is not None else (int(c[0]), 0))
    lengths = [len(r["conditions"]) for r in rules]
    return {
        "features_used_idx": feats, "features_used_names": [nm(j) for j in feats],
        "root_feature_idx": root_feat, "root_feature_names": [nm(j) for j in root_feat],
        "split_signatures": [list(s) for s in sorted(splits)],
        "rule_count": len(rules), "average_rule_literals": float(np.mean(lengths)) if lengths else 0.0,
        "max_rule_literals": int(max(lengths)) if lengths else 0, "node_cap_reached": False, "max_nodes": 0,
    }


def structure_columns(info: Dict[str, Any]) -> Dict[str, Any]:
    """Colunas planas (listas serializadas em JSON: seguras em CSV e reconstruíveis) para a tabela de resultados."""
    return {
        "features_used_names": json.dumps(info["features_used_names"]), "root_feature": json.dumps(info["root_feature_names"]),
        "split_signatures": json.dumps(info["split_signatures"]), "rule_count": int(info["rule_count"]),
        "average_rule_literals": float(info["average_rule_literals"]), "max_rule_literals": int(info["max_rule_literals"]),
        "node_cap_reached": bool(info["node_cap_reached"]), "tree_max_nodes": int(info["max_nodes"]),
    }
