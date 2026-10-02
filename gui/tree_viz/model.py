"""Modelo visual de árvores (SOMENTE LEITURA) para C4.5, TREPAN Original/Reloaded e árvores estilo sklearn.

``build_visualization_model`` lê a árvore científica e produz ``VizNode`` independentes. Nada
aqui escreve na árvore: abreviar nomes ou arredondar limiares para display altera apenas os
campos ``*_display`` do objeto visual; ``feature_full`` / ``threshold_full`` guardam o valor real.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from gui.tree_viz.labels import (
    display_class_name, format_threshold, full_threshold, is_ontology_feature,
    make_display_feature_name, short_class_name,
)
from gui.tree_viz.strings import tr

SUPPORTED_SPLIT_TYPES = ("simple", "m_of_n", "categorical")


class TreeVisualizationError(Exception):
    """Erro de validação/renderização com código (TREE_VISUALIZATION_ERROR / INVALID_TREE_CYCLE ...)."""

    def __init__(self, code: str, message: str = ""):
        super().__init__(f"{tr('viz_error')} [{code}] {message}".strip())
        self.code = code
        self.detail = message


@dataclass
class VizEdge:
    parent_id: int
    child_id: int
    label: str                 # texto compacto desenhado junto à aresta
    full_label: str            # condição completa (com nome real da feature e limiar real)
    branch: str                # "left" | "right" | "child"
    samples: Optional[int] = None


@dataclass
class VizNode:
    node_id: int
    depth: int
    is_leaf: bool
    parent_id: Optional[int] = None
    samples: Optional[int] = None
    class_prediction: Optional[str] = None
    class_distribution: Dict[str, float] = field(default_factory=dict)
    distribution_kind: str = "counts"            # counts | probability
    split_type: Optional[str] = None             # simple | m_of_n | categorical
    feature_full: Optional[str] = None
    feature_display: Optional[str] = None
    threshold_full: Optional[float] = None
    threshold_display: Optional[str] = None
    conditions: List[dict] = field(default_factory=list)
    m: Optional[int] = None
    n: Optional[int] = None
    children: List[int] = field(default_factory=list)   # ordem visual esquerda -> direita
    edges: List[VizEdge] = field(default_factory=list)
    is_ontology_feature: bool = False
    is_semantic_split: bool = False
    semantic: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- métricas de incerteza (definição única e explícita) ----------------------------
    @property
    def total(self) -> float:
        return float(sum(self.class_distribution.values()))

    @property
    def confidence(self) -> float:
        """Probabilidade da classe dominante no nó."""
        t = self.total
        return float(max(self.class_distribution.values()) / t) if t > 0 else 0.0

    @property
    def uncertainty(self) -> float:
        """Entropia normalizada [0,1] da distribuição de classes do nó (0 puro, 1 máxima mistura)."""
        t = self.total
        k = len(self.class_distribution)
        if t <= 0 or k < 2:
            return 0.0
        p = [v / t for v in self.class_distribution.values() if v > 0]
        return float(-sum(x * math.log2(x) for x in p) / math.log2(k))

    # ---- texto desenhado dentro do círculo (compacto) ------------------------------------
    def display_lines(self, show_uncertainty: bool = False, scientific: bool = False, show_id: bool = False) -> List[str]:
        if self.is_leaf:
            lines = [short_class_name(self.class_prediction or "?", 9)]
        elif self.split_type == "m_of_n":
            lines = ["m-of-n", f"{self.m}/{self.n}"]
        else:
            lines = [self.feature_display or "?"]
        if self.samples is not None and self.split_type != "m_of_n":
            lines.append(f"({self.samples})")
        elif self.samples is not None and scientific:
            lines.append(f"({self.samples})")
        if show_uncertainty:
            lines.append(f"{self.confidence * 100:.0f}%" if self.is_leaf else f"H {self.uncertainty:.2f}")
        if show_id:
            lines.insert(0, f"N{self.node_id}")
        return lines


@dataclass
class TreeVisualizationModel:
    nodes: Dict[int, VizNode]
    root_id: int
    algorithm: str
    class_names: List[str]
    feature_names: List[str]
    logical_node_count: int
    source_kind: str

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def depth(self) -> int:
        return max((n.depth for n in self.nodes.values()), default=0)

    def preorder(self, start: Optional[int] = None, hidden: Optional[set] = None) -> List[int]:
        out, stack = [], [self.root_id if start is None else start]
        hidden = hidden or set()
        while stack:
            nid = stack.pop()
            out.append(nid)
            if nid in hidden:
                continue
            stack.extend(reversed(self.nodes[nid].children))
        return out

    def path_to(self, node_id: int) -> List[int]:
        path, cur = [], node_id
        while cur is not None:
            path.append(cur)
            cur = self.nodes[cur].parent_id
        return list(reversed(path))

    def subtree_size(self, node_id: int) -> int:
        return len(self.preorder(node_id))

    def edge(self, parent_id: int, child_id: int) -> Optional[VizEdge]:
        for e in self.nodes[parent_id].edges:
            if e.child_id == child_id:
                return e
        return None

    def class_labels(self) -> List[str]:
        """Rótulos de classe na ordem do dataset, depois quaisquer outros presentes."""
        labels = list(dict.fromkeys(str(c) for c in self.class_names))
        for n in self.nodes.values():
            for c in ([n.class_prediction] if n.class_prediction else []) + list(n.class_distribution):
                if str(c) not in labels:
                    labels.append(str(c))
        return labels

    def to_dict(self) -> dict:
        def node_dict(n: VizNode) -> dict:
            return {
                "node_id": n.node_id, "depth": n.depth, "is_leaf": n.is_leaf, "parent_id": n.parent_id,
                "samples": n.samples, "class_prediction": n.class_prediction,
                "class_distribution": n.class_distribution, "distribution_kind": n.distribution_kind,
                "split_type": n.split_type, "feature_full": n.feature_full,
                "threshold_full": n.threshold_full, "conditions": n.conditions, "m": n.m, "n": n.n,
                "children": n.children, "edges": [e.__dict__ for e in n.edges],
                "semantic": n.semantic, "metadata": n.metadata,
            }
        return {
            "algorithm": self.algorithm, "source_kind": self.source_kind, "root_id": self.root_id,
            "logical_node_count": self.logical_node_count, "class_names": self.class_names,
            "feature_names": self.feature_names,
            "nodes": [node_dict(self.nodes[i]) for i in self.preorder()],
        }


# =============================================================================== helpers
def _jsonable(v: Any) -> Any:
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def _feature_name(names: Sequence[str], idx: int) -> str:
    return str(names[idx]) if 0 <= idx < len(names) else f"feature_{idx}"


def _classes_from_tree(tree) -> List[Any]:
    return list(np.asarray(getattr(tree, "classes_", [])).tolist())


def _label_for(raw, class_names) -> str:
    return display_class_name(raw, class_names)


def _check_threshold(value, node_id) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise TreeVisualizationError("INVALID_THRESHOLD", f"nó {node_id}: limiar não numérico {value!r}")
    if not math.isfinite(v):
        raise TreeVisualizationError("INVALID_THRESHOLD", f"nó {node_id}: limiar não finito {value!r}")
    return v


def source_node_count(tree) -> int:
    """Contagem lógica de nós, feita de forma INDEPENDENTE do adaptador (para comparar com o renderizado)."""
    root = getattr(tree, "root_", None)
    if root is not None:
        seen, stack, count = set(), [root], 0
        while stack:
            n = stack.pop()
            if id(n) in seen:
                raise TreeVisualizationError("INVALID_TREE_CYCLE", "ciclo na estrutura da árvore")
            seen.add(id(n))
            count += 1
            if hasattr(n, "children") and isinstance(getattr(n, "children"), dict):
                stack.extend(n.children.values())
            else:
                for attr in ("false_child", "true_child"):
                    c = getattr(n, attr, None)
                    if c is not None:
                        stack.append(c)
        return count
    t = getattr(tree, "tree_", None)
    if t is not None:
        return int(len(t.children_left))
    raise TreeVisualizationError("UNSUPPORTED_TREE", f"tipo de árvore não suportado: {type(tree).__name__}")


def tree_signature(tree) -> str:
    """Assinatura estável (hash) da estrutura científica: usada para provar que o render não a altera."""
    return hashlib.sha256(json.dumps(_signature_payload(tree), sort_keys=True, default=str).encode()).hexdigest()


def _signature_payload(tree) -> Any:
    root = getattr(tree, "root_", None)
    if root is not None and hasattr(root, "false_child"):
        def walk(n):
            d = {"id": int(n.node_id), "depth": int(n.depth), "pred": _jsonable(n.prediction),
                 "dist": [repr(float(x)) for x in np.asarray(n.distribution).tolist()]}
            if not n.is_leaf:
                d["m"] = int(n.test.m)
                d["lits"] = [(int(l.feature), repr(float(l.threshold)), bool(l.greater)) for l in n.test.literals]
                d["f"] = walk(n.false_child); d["t"] = walk(n.true_child)
            return d
        return {"kind": "trepan", "classes": _jsonable(_classes_from_tree(tree)), "root": walk(root)}
    if root is not None and isinstance(getattr(root, "children", None), dict):
        def walk(n):
            d = {"id": int(getattr(n, "node_id", -1)), "depth": int(n.depth),
                 "counts": [repr(float(x)) for x in np.asarray(n.class_counts).tolist()],
                 "pred": int(n.prediction_index)}
            if not n.is_leaf:
                d.update(feature=int(n.feature), kind=n.kind, thr=None if n.threshold is None else repr(float(n.threshold)),
                         ch={str(k): walk(v) for k, v in n.children.items()})
            return d
        return {"kind": "c45", "classes": _jsonable(_classes_from_tree(tree)), "root": walk(root)}
    t = tree.tree_
    return {"kind": "sk", "l": np.asarray(t.children_left).tolist(), "r": np.asarray(t.children_right).tolist(),
            "f": np.asarray(t.feature).tolist(), "t": [repr(float(x)) for x in np.asarray(t.threshold).tolist()],
            "v": [repr(float(x)) for x in np.asarray(t.value).ravel().tolist()]}


# =============================================================================== adapters
def build_visualization_model(tree, feature_names=None, class_names=None, algorithm: Optional[str] = None) -> TreeVisualizationModel:
    """Lê a árvore (sem a modificar) e devolve o modelo visual validado."""
    if tree is None:
        raise TreeVisualizationError("NO_TREE", tr("no_tree"))
    feature_names = [str(f) for f in (feature_names or [])]
    class_names = [str(c) for c in (class_names or [])]
    logical = source_node_count(tree)
    root = getattr(tree, "root_", None)
    if root is not None and hasattr(root, "false_child") and hasattr(root, "test"):
        model = _from_trepan(tree, feature_names, class_names, algorithm or "TREPAN")
    elif root is not None and isinstance(getattr(root, "children", None), dict):
        model = _from_c45(tree, feature_names, class_names, algorithm or "C4.5")
    elif getattr(tree, "tree_", None) is not None:
        model = _from_sklearn_like(tree, feature_names, class_names, algorithm or "Tree")
    else:
        raise TreeVisualizationError("UNSUPPORTED_TREE", f"tipo de árvore não suportado: {type(tree).__name__}")
    model.logical_node_count = logical
    validate_model(model)
    return model


def validate_model(model: TreeVisualizationModel) -> None:
    if model.root_id not in model.nodes:
        raise TreeVisualizationError("MISSING_ROOT", "a raiz não existe no modelo visual")
    seen, stack = set(), [model.root_id]
    while stack:
        nid = stack.pop()
        if nid in seen:
            raise TreeVisualizationError("INVALID_TREE_CYCLE", f"nó {nid} visitado duas vezes")
        seen.add(nid)
        n = model.nodes[nid]
        if not n.is_leaf and n.split_type not in SUPPORTED_SPLIT_TYPES:
            raise TreeVisualizationError("UNKNOWN_SPLIT_TYPE", f"nó {nid}: {n.split_type!r}")
        if not n.is_leaf and not n.children:
            raise TreeVisualizationError("MISSING_CHILD", f"nó interno {nid} sem filhos")
        for c in n.children:
            if c not in model.nodes:
                raise TreeVisualizationError("MISSING_CHILD", f"nó {nid} refere filho inexistente {c}")
            stack.append(c)
    if len(seen) != len(model.nodes):
        raise TreeVisualizationError("UNREACHABLE_NODES", f"{len(model.nodes) - len(seen)} nós inalcançáveis")


def _semantic_rows(tree) -> Dict[int, dict]:
    rows = getattr(tree, "semantic_split_audit_", None) or []
    return {int(r["node_id"]): r for r in rows if "node_id" in r}


def _from_trepan(tree, feature_names, class_names, algorithm) -> TreeVisualizationModel:
    classes = _classes_from_tree(tree)
    labels = [_label_for(c, class_names) for c in classes]
    sem_rows = _semantic_rows(tree)
    nodes: Dict[int, VizNode] = {}
    seen_obj: set = set()

    def pred_label(prediction) -> str:
        return _label_for(prediction, class_names)

    def build(src, parent_id) -> VizNode:
        if id(src) in seen_obj:
            raise TreeVisualizationError("INVALID_TREE_CYCLE", f"nó {src.node_id} referido por mais de um pai")
        seen_obj.add(id(src))
        nid = int(src.node_id)
        if nid in nodes:
            raise TreeVisualizationError("DUPLICATE_NODE_ID", f"id {nid} duplicado")
        dist = np.asarray(src.distribution, dtype=float)
        v = VizNode(
            node_id=nid, depth=int(src.depth), is_leaf=bool(src.is_leaf), parent_id=parent_id,
            samples=int(len(src.real_y)), class_prediction=pred_label(src.prediction),
            class_distribution={(labels[i] if i < len(labels) else str(i)): float(p) for i, p in enumerate(dist)},
            distribution_kind="probability",
        )
        meta = {"reach": float(src.reach), "node_fidelity": float(src.fidelity)}
        stats = dict(getattr(src, "stats", {}) or {})
        for k in ("real_samples", "synthetic_samples", "effective_samples", "queries_used", "queries_requested"):
            if k in stats:
                meta[k] = stats[k]
        if getattr(src, "stop_reason", None):
            meta["stop_reason"] = src.stop_reason
            meta["stop_detail"] = _jsonable(getattr(src, "stop_detail", {}))
        v.metadata = meta
        nodes[nid] = v
        if src.is_leaf:
            return v
        test = src.test
        lits = list(test.literals)
        simple = len(lits) == 1 and int(test.m) == 1
        conds = []
        for lit in lits:
            thr = _check_threshold(lit.threshold, nid)
            fname = _feature_name(feature_names, int(lit.feature))
            op = ">" if lit.greater else "≤"
            conds.append({"feature_full": fname, "feature_display": make_display_feature_name(fname),
                          "op": op, "threshold_full": thr,
                          "text_full": f"{fname} {op} {full_threshold(thr)}",
                          "text_display": f"{make_display_feature_name(fname)} {op} {format_threshold(thr)}"})
        v.conditions = conds
        v.m, v.n = int(test.m), len(lits)
        f_child, t_child = src.false_child, src.true_child
        if simple:
            lit = lits[0]
            c0 = conds[0]
            v.split_type = "simple"
            v.feature_full, v.feature_display = c0["feature_full"], c0["feature_display"]
            v.threshold_full, v.threshold_display = c0["threshold_full"], format_threshold(c0["threshold_full"])
            v.is_ontology_feature = is_ontology_feature(v.feature_full)
            le_text, gt_text = f"≤ {v.threshold_display}", f"> {v.threshold_display}"
            le_full = f"{v.feature_full} ≤ {full_threshold(v.threshold_full)}"
            gt_full = f"{v.feature_full} > {full_threshold(v.threshold_full)}"
            # convenção visual única: ramo "≤" à esquerda, ramo ">" à direita (independente do sinal do literal)
            le_child, gt_child = (f_child, t_child) if lit.greater else (t_child, f_child)
            pairs = [(le_child, le_text, le_full, "left"), (gt_child, gt_text, gt_full, "right")]
        else:
            v.split_type = "m_of_n"
            v.is_ontology_feature = any(is_ontology_feature(c["feature_full"]) for c in conds)
            v.metadata["m_of_n_text"] = test.text(feature_names)
            pairs = [(f_child, tr("no"), f"NOT {v.m}-of-{v.n}", "left"),
                     (t_child, tr("yes"), f"{v.m}-of-{v.n}", "right")]
        row = sem_rows.get(nid)
        if row is not None:
            bonus = float(row.get("semantic_bonus", 0.0) or 0.0)
            influenced = bool(row.get("ontology_influenced", False))
            v.is_semantic_split = bool(bonus > 1e-12 or influenced)
            v.semantic = {
                "base_score": float(row.get("information_gain", 0.0)),
                "semantic_bonus": bonus,
                "final_score": float(row.get("selection_score", 0.0)),
                "semantic_groups": list(row.get("semantic_groups", []) or []),
                "semantic_features": [_feature_name(feature_names, int(i)) for i in (row.get("semantic_features") or [])],
                "ontology_influenced": influenced,
                "decision_changed": bool(row.get("decision_changed", False)),
            }
        elif nid in {int(r.get("node_id", -1)) for r in (getattr(tree, "split_audit_", None) or [])}:
            for r in tree.split_audit_:
                if int(r.get("node_id", -1)) == nid:
                    v.metadata["information_gain"] = float(r.get("information_gain", 0.0))
        for child_src, label, full, branch in pairs:
            child = build(child_src, nid)
            v.children.append(child.node_id)
            v.edges.append(VizEdge(nid, child.node_id, label, full, branch, child.samples))
        return v

    root_v = build(tree.root_, None)
    return TreeVisualizationModel(nodes, root_v.node_id, algorithm, class_names, feature_names, len(nodes), "trepan")


def _from_c45(tree, feature_names, class_names, algorithm) -> TreeVisualizationModel:
    classes = _classes_from_tree(tree)
    labels = [_label_for(c, class_names) for c in classes]
    nodes: Dict[int, VizNode] = {}
    seen_obj: set = set()
    counter = {"next": 0}
    cat_labels = getattr(tree, "category_labels_", {}) or {}
    ids = []

    def collect(n):
        ids.append(getattr(n, "node_id", -1))
        for c in n.children.values():
            collect(c)
    collect(tree.root_)
    use_native_ids = len(set(ids)) == len(ids) and min(ids) >= 0

    def build(src, parent_id) -> VizNode:
        if id(src) in seen_obj:
            raise TreeVisualizationError("INVALID_TREE_CYCLE", "nó referido por mais de um pai")
        seen_obj.add(id(src))
        nid = int(src.node_id) if use_native_ids else counter["next"]
        counter["next"] += 1
        if nid in nodes:
            raise TreeVisualizationError("DUPLICATE_NODE_ID", f"id {nid} duplicado")
        counts = np.asarray(src.class_counts, dtype=float)
        v = VizNode(
            node_id=nid, depth=int(src.depth), is_leaf=bool(src.is_leaf), parent_id=parent_id,
            samples=int(round(float(src.n_samples))),
            class_prediction=(labels[int(src.prediction_index)] if int(src.prediction_index) < len(labels) else str(src.prediction_index)),
            class_distribution={(labels[i] if i < len(labels) else str(i)): float(c) for i, c in enumerate(counts)},
            distribution_kind="counts",
        )
        if getattr(src, "stop_reason", None):
            v.metadata["stop_reason"] = src.stop_reason
        nodes[nid] = v
        if src.is_leaf:
            return v
        feat = int(src.feature)
        fname = _feature_name(feature_names, feat)
        v.feature_full, v.feature_display = fname, make_display_feature_name(fname)
        v.is_ontology_feature = is_ontology_feature(fname)
        v.metadata.update(gain=float(src.gain), gain_ratio=float(src.gain_ratio))
        if src.kind == "numeric":
            thr = _check_threshold(src.threshold, nid)
            v.split_type, v.threshold_full, v.threshold_display = "simple", thr, format_threshold(thr)
            v.conditions = [{"feature_full": fname, "feature_display": v.feature_display, "op": "≤",
                             "threshold_full": thr, "text_full": f"{fname} ≤ {full_threshold(thr)}",
                             "text_display": f"{v.feature_display} ≤ {v.threshold_display}"}]
            pairs = [(src.children["left"], f"≤ {v.threshold_display}", f"{fname} ≤ {full_threshold(thr)}", "left"),
                     (src.children["right"], f"> {v.threshold_display}", f"{fname} > {full_threshold(thr)}", "right")]
        else:
            v.split_type = "categorical"
            labmap = cat_labels.get(feat, {})
            pairs = []
            for key in sorted(src.children):
                txt = str(labmap.get(key, key))
                pairs.append((src.children[key], f"= {txt}", f"{fname} = {txt}", "child"))
        for child_src, label, full, branch in pairs:
            child = build(child_src, nid)
            v.children.append(child.node_id)
            v.edges.append(VizEdge(nid, child.node_id, label, full, branch, child.samples))
        return v

    root_v = build(tree.root_, None)
    return TreeVisualizationModel(nodes, root_v.node_id, algorithm, class_names, feature_names, len(nodes), "c45")


def _from_sklearn_like(tree, feature_names, class_names, algorithm) -> TreeVisualizationModel:
    t = tree.tree_
    left, right = np.asarray(t.children_left), np.asarray(t.children_right)
    classes = _classes_from_tree(tree)
    labels = [_label_for(c, class_names) for c in classes]
    nodes: Dict[int, VizNode] = {}
    visiting: set = set()

    def build(i: int, parent_id, depth) -> VizNode:
        if i in visiting or i in nodes:
            raise TreeVisualizationError("INVALID_TREE_CYCLE", f"nó {i} alcançado mais de uma vez")
        visiting.add(i)
        value = np.asarray(t.value[i], dtype=float).reshape(-1)
        n_samples = int(np.asarray(t.n_node_samples)[i]) if hasattr(t, "n_node_samples") else None
        total = float(value.sum())
        # sklearn >= 1.4 guarda frações em tree_.value: converte para contagens
        counts = value * n_samples if (n_samples is not None and total > 0 and abs(total - 1.0) < 1e-6) else value
        names = labels if labels else [_label_for(j, class_names) for j in range(len(counts))]
        is_leaf = int(left[i]) == -1 and int(right[i]) == -1
        dom = int(np.argmax(counts)) if len(counts) else 0
        v = VizNode(
            node_id=int(i), depth=depth, is_leaf=is_leaf, parent_id=parent_id,
            samples=n_samples, class_prediction=(names[dom] if dom < len(names) else str(dom)),
            class_distribution={(names[j] if j < len(names) else str(j)): float(c) for j, c in enumerate(counts)},
        )
        nodes[i] = v
        if not is_leaf:
            feat = int(np.asarray(t.feature)[i])
            thr = _check_threshold(np.asarray(t.threshold)[i], i)
            fname = _feature_name(feature_names, feat)
            v.split_type, v.feature_full, v.feature_display = "simple", fname, make_display_feature_name(fname)
            v.threshold_full, v.threshold_display = thr, format_threshold(thr)
            v.is_ontology_feature = is_ontology_feature(fname)
            v.conditions = [{"feature_full": fname, "feature_display": v.feature_display, "op": "≤",
                             "threshold_full": thr, "text_full": f"{fname} ≤ {full_threshold(thr)}",
                             "text_display": f"{v.feature_display} ≤ {v.threshold_display}"}]
            for ci, label, full, br in ((int(left[i]), f"≤ {v.threshold_display}", f"{fname} ≤ {full_threshold(thr)}", "left"),
                                        (int(right[i]), f"> {v.threshold_display}", f"{fname} > {full_threshold(thr)}", "right")):
                if ci < 0:
                    raise TreeVisualizationError("MISSING_CHILD", f"nó {i}: filho em falta")
                child = build(ci, i, depth + 1)
                v.children.append(child.node_id)
                v.edges.append(VizEdge(i, child.node_id, label, full, br, child.samples))
        visiting.discard(i)
        return v

    build(0, None, 0)
    return TreeVisualizationModel(nodes, 0, algorithm, class_names, feature_names, len(nodes), "sklearn")
