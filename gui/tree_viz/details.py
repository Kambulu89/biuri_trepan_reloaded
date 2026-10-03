"""Textos de detalhe (tooltip, painel de nó/aresta, caminho de regra). Só leitura, sem Qt."""
from __future__ import annotations

from typing import List, Optional

from gui.tree_viz.labels import format_threshold, full_threshold
from gui.tree_viz.model import TreeVisualizationModel, VizNode
from gui.tree_viz.strings import tr


def _dist_lines(node: VizNode) -> List[str]:
    total = node.total
    out = []
    for cls, v in node.class_distribution.items():
        if node.distribution_kind == "probability":
            out.append(f"  {cls}: {v:.4f}")
        else:
            pct = f" ({100 * v / total:.1f}%)" if total > 0 else ""
            out.append(f"  {cls}: {v:g}{pct}")
    return out


def mofn_block(node: VizNode, indent: str = "") -> List[str]:
    lines = [f"{indent}{node.m}-of-{node.n}:"]
    for i, c in enumerate(node.conditions, 1):
        lines.append(f"{indent}  {i}. {c['text_full']}")
    return lines


def node_details_text(model: TreeVisualizationModel, node_id: int) -> str:
    n = model.nodes[node_id]
    L: List[str] = [tr("node_details"), "=" * len(tr("node_details")),
                    f"{tr('node_id')}: {n.node_id}", f"{tr('depth')}: {n.depth}",
                    f"{tr('type')}: {tr('leaf') if n.is_leaf else tr('internal')}"]
    if n.samples is not None:
        L.append(f"{tr('samples')}: {n.samples}")
    if not n.is_leaf:
        L += ["", tr("split"), "-" * len(tr("split")), f"{tr('split_type')}: {tr(n.split_type)}"]
        if n.split_type == "m_of_n":
            L += [f"{tr('mofn_details')}: m = {n.m}, n = {n.n}", f"{tr('conditions')}:"]
            L += [f"  {i}. {c['text_full']}" for i, c in enumerate(n.conditions, 1)]
            L.append(tr("true_when", m=n.m, n=n.n))
        else:
            L += [f"{tr('feature_full')}: {n.feature_full}"]
            if n.threshold_full is not None:
                L.append(f"{tr('threshold')}: {full_threshold(n.threshold_full)}  (display: {n.threshold_display})")
            for e in n.edges:
                L.append(f"  → {e.full_label}   [{e.samples} {tr('samples').lower()}]")
    L += ["", tr("prediction"), "-" * len(tr("prediction")),
          f"{tr('dominant_class')}: {n.class_prediction}", f"{tr('class_distribution')}:"] + _dist_lines(n)
    L += [f"{tr('confidence')}: {n.confidence:.4f}", f"{tr('entropy')}: {n.uncertainty:.4f}", tr("uncertainty_def")]
    md = n.metadata
    trepan_keys = [k for k in ("real_samples", "synthetic_samples", "effective_samples", "queries_used") if k in md]
    if trepan_keys or "stop_reason" in md:
        L += ["", tr("trepan_info"), "-" * len(tr("trepan_info"))]
        names = {"real_samples": "real_samples", "synthetic_samples": "synthetic_samples",
                 "effective_samples": "effective_samples", "queries_used": "queries"}
        for k in trepan_keys:
            L.append(f"{tr(names[k])}: {md[k]}")
        if "stop_reason" in md:
            L.append(f"{tr('stop_reason')}: {md['stop_reason']}  {md.get('stop_detail') or ''}".rstrip())
    if "gain_ratio" in md:
        L += ["", "C4.5", "----", f"gain: {md['gain']:.6f}", f"gain_ratio: {md['gain_ratio']:.6f}"]
    if n.semantic:
        s = n.semantic
        L += ["", tr("reloaded_info"), "-" * len(tr("reloaded_info")),
              f"{tr('base_score')}: {s['base_score']:.6f}", f"{tr('semantic_score')}: {s['semantic_bonus']:.6f}",
              f"{tr('final_score')}: {s['final_score']:.6f}"]
        if s.get("semantic_groups"):
            L.append(f"{tr('family')}: {', '.join(map(str, s['semantic_groups']))}")
        if s.get("semantic_features"):
            L.append(f"{tr('semantic_features')}: {', '.join(s['semantic_features'])}")
    if n.is_ontology_feature and n.feature_full:
        L += ["", f"{tr('feature_full')}: {n.feature_full}"]
    if not n.is_leaf or n.parent_id is not None:
        L += ["", rule_path_text(model, node_id)]
    return "\n".join(L)


def edge_details_text(model: TreeVisualizationModel, parent_id: int, child_id: int) -> str:
    e = model.edge(parent_id, child_id)
    if e is None:
        return ""
    p = model.nodes[parent_id]
    L = [tr("edge_details"), "=" * len(tr("edge_details")), f"{tr('parent')}: {parent_id}", f"{tr('child')}: {child_id}",
         f"{tr('condition')}: {e.full_label}", f"{tr('branch')}: {e.branch}"]
    if e.samples is not None:
        L.append(f"{tr('samples_child')}: {e.samples}")
    if p.split_type == "m_of_n":
        L += [""] + mofn_block(p)
    return "\n".join(L)


def rule_path_text(model: TreeVisualizationModel, node_id: int) -> str:
    """Caminho raiz -> nó. m-of-n permanece m-of-n (nunca convertido em regra simples)."""
    path = model.path_to(node_id)
    conds: List[List[str]] = []
    for parent_id, child_id in zip(path, path[1:]):
        p = model.nodes[parent_id]
        e = model.edge(parent_id, child_id)
        if p.split_type == "m_of_n":
            neg = "NO " if e.branch == "left" else ""
            conds.append([f"{neg}{p.m}-of-{p.n}:"] + [f"    {i}. {c['text_full']}" for i, c in enumerate(p.conditions, 1)])
        else:
            conds.append([e.full_label])
    n = model.nodes[node_id]
    head = f"{tr('rule_path')}"
    if not conds:
        return f"{head}\n{'-' * len(head)}\n(raíz)"
    out = [head, "-" * len(head)]
    for i, block in enumerate(conds):
        out.append(f"{tr('if') if i == 0 else tr('and')} {block[0]}")
        out += block[1:]
    if n.is_leaf:
        out.append(f"{tr('then')} {n.class_prediction}   ({n.samples} {tr('samples').lower()})")
    return "\n".join(out)


def node_tooltip(model: TreeVisualizationModel, node_id: int) -> str:
    n = model.nodes[node_id]
    if n.is_leaf:
        L = [f"{tr('leaf')} {n.node_id}", f"Prediction: {n.class_prediction}"]
        if n.samples is not None:
            L.append(f"{tr('samples')}: {n.samples}")
        L.append(f"{tr('class_distribution')}:")
        L += _dist_lines(n)
        return "\n".join(L)
    L = [f"{tr('node')} {n.node_id}  ({tr(n.split_type)})"]
    if n.split_type == "m_of_n":
        L.append(f"m-of-n: {n.m}/{n.n}")
        L += [f"  {i}. {c['text_full']}" for i, c in enumerate(n.conditions, 1)]
    else:
        L.append(f"Feature: {n.feature_full}")
        if n.threshold_full is not None:
            L.append(f"{tr('threshold')}: {full_threshold(n.threshold_full)}")
    if n.samples is not None:
        L.append(f"{tr('samples')}: {n.samples}")
    L.append(f"{tr('depth')}: {n.depth}")
    if n.node_id == model.root_id:
        L.append("Root node")
    if n.semantic and n.is_semantic_split:
        L.append(f"{tr('semantic_score')}: {n.semantic['semantic_bonus']:.4f}")
    return "\n".join(L)
