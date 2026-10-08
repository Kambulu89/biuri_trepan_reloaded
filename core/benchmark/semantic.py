"""Fornecedores de estrutura semântica: sem semântica, OWL real, OWL SHUFFLED (controlo negativo) e grupos.

Todos ajustam-se SÓ ao treino e expõem o mesmo contrato (``SemanticContext``), de modo que as
arquitecturas Reloaded são idênticas e a ÚNICA diferença entre os braços é o conteúdo semântico.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

JUNK_GROUPS = ("DatatypeProperty", "ObjectProperty", "AnnotationProperty")


@dataclass
class SemanticContext:
    source: str                                  # none | real | shuffled
    provider: str
    available: bool
    base_names: List[str]
    enriched_names: List[str]
    orig_idx: List[int]
    onto_idx: List[int]
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None
    weights: Optional[np.ndarray] = None
    groups: Optional[List[Optional[str]]] = None
    relatedness: Optional[np.ndarray] = None
    depths: Optional[np.ndarray] = None
    ontology_valid: Optional[bool] = None
    reason: str = ""
    semantic_time: float = 0.0
    reasoner_time: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_features(self) -> int:
        return len(self.enriched_names)

    def enrich(self, X) -> np.ndarray:
        if not self.available or self.transform is None:
            raise RuntimeError("Contexto semântico indisponível.")
        return np.asarray(self.transform(np.asarray(X, dtype=float)), dtype=float)

    def query_projector(self) -> Callable[[np.ndarray], np.ndarray]:
        """Re-calcula as features derivadas a partir das colunas originais das membership queries."""
        idx = list(self.orig_idx)
        return lambda rows: self.enrich(np.asarray(rows, dtype=float)[:, idx])

    def structure_signature(self) -> str:
        h = hashlib.sha256()
        for part in (self.weights, self.relatedness, self.depths):
            h.update(b"-" if part is None else np.ascontiguousarray(part, dtype=float).tobytes())
        h.update(repr(self.groups).encode())
        return h.hexdigest()[:16]


def structure_counts(ctx: "SemanticContext") -> Dict[str, Any]:
    """Contagens estruturais da semântica (classes/grupos, relações, features derivadas) para o registo do controlo negativo."""
    groups = [g for g in (ctx.groups or []) if g is not None]
    rel = np.asarray(ctx.relatedness) if ctx.relatedness is not None else np.zeros((0, 0))
    off = rel - np.diag(np.diag(rel)) if rel.size else rel
    return {"n_semantic_classes": int(len(set(groups))), "n_grouped_features": int(len(groups)),
            "n_relations": int(np.count_nonzero(np.triu(off, 1))) if rel.size else 0,
            "relatedness_sum": float(np.triu(off, 1).sum()) if rel.size else 0.0,
            "n_semantic_features": int(len(ctx.onto_idx)),
            "n_nonunit_weights": int(np.count_nonzero(np.asarray(ctx.weights) != 1.0)) if ctx.weights is not None else 0}


class NoSemanticProvider:
    name = "none"

    def build(self, X_train, feature_names, seed: int = 0) -> SemanticContext:
        names = [str(n) for n in feature_names]
        return SemanticContext("none", self.name, False, names, names, list(range(len(names))), [],
                               reason="sem ontologia: semantic_available=false")


class GroupSemanticProvider:
    """Estrutura semântica sem OWL: grupos de features -> agregados padronizados (ajustados no treino).

    Serve de ontologia programática mínima (e de teste sem reasoner). Não conhece nenhum dataset."""
    name = "groups"

    def __init__(self, groups: Dict[str, Sequence[str]], onto_weight: float = 1.35):
        self.groups = {k: [str(f) for f in v] for k, v in groups.items()}
        self.onto_weight = float(onto_weight)

    def build(self, X_train, feature_names, seed: int = 0) -> SemanticContext:
        t0 = time.perf_counter()
        names = [str(n) for n in feature_names]
        pos = {n: i for i, n in enumerate(names)}
        unknown = [f for fs in self.groups.values() for f in fs if f not in pos]
        if unknown:
            return SemanticContext("none", self.name, False, names, names, list(range(len(names))), [],
                                   ontology_valid=False, reason=f"features desconhecidas nos grupos: {unknown[:3]}")
        X = np.asarray(X_train, dtype=float)
        gnames = list(self.groups)
        idxs = {g: [pos[f] for f in self.groups[g]] for g in gnames}
        mu = {g: X[:, idxs[g]].mean(axis=0) for g in gnames}
        sd = {g: np.where(X[:, idxs[g]].std(axis=0) > 1e-12, X[:, idxs[g]].std(axis=0), 1.0) for g in gnames}

        def transform(Z):
            Z = np.asarray(Z, dtype=float)
            cols = [((Z[:, idxs[g]] - mu[g]) / sd[g]).mean(axis=1, keepdims=True) for g in gnames]
            return np.hstack([Z] + cols)

        n_orig = len(names)
        enriched = names + [f"onto_{g}_aggregate" for g in gnames]
        feat_group: List[Optional[str]] = [None] * n_orig
        for g in gnames:
            for i in idxs[g]:
                feat_group[i] = g
        groups = feat_group + list(gnames)
        n = len(enriched)
        rel = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if groups[i] is not None and groups[i] == groups[j]:
                    rel[i, j] = rel[j, i] = 1.0
        weights = np.ones(n)
        weights[n_orig:] = self.onto_weight
        depths = np.array([1.0 if g is not None else 0.0 for g in groups])
        return SemanticContext("real", self.name, True, names, enriched, list(range(n_orig)), list(range(n_orig, n)),
                               transform, weights, groups, rel, depths, ontology_valid=True,
                               semantic_time=time.perf_counter() - t0, info={"n_groups": len(gnames), "mapped_feature_count": sum(1 for g in feat_group if g is not None),
                                     "unmapped_feature_count": sum(1 for g in feat_group if g is None), "reasoning_applied": False,
                                     "ontology_hash": hashlib.sha256(repr(sorted(self.groups.items())).encode()).hexdigest()})


class OwlSemanticProvider:
    """OWL real: valida (reasoner + quality gate), gera features derivadas (OntologyProcessor, ajustado no treino)
    e a estrutura semântica (grupos, relatedness, depths) a partir do grafo da ontologia."""
    name = "owl_real"

    def __init__(self, ontology_path: str, reasoner_engine: str = "hermit", onto_weight: float = 1.35):
        self.ontology_path = str(ontology_path)
        self.reasoner_engine = reasoner_engine
        self.onto_weight = float(onto_weight)

    def build(self, X_train, feature_names, seed: int = 0) -> SemanticContext:
        names = [str(n) for n in feature_names]
        n_orig = len(names)
        empty = lambda reason, valid: SemanticContext("none", self.name, False, names, names, list(range(n_orig)), [],
                                                      ontology_valid=valid, reason=reason)
        try:
            from owlready2 import get_ontology
            from core.ontology_processor import OntologyProcessor
            from core.ontology_quality import OntologyQualityGate
            from core.ontology_reasoner import run_owl_reasoner
            from core.ontology_semantic_graph import OntologySemanticGraph
        except ImportError as exc:
            return empty(f"dependência OWL em falta: {exc}", None)
        t0 = time.perf_counter()
        try:
            from core.owl_runtime import load_ontology_isolated
            onto = load_ontology_isolated(self.ontology_path)       # mundo OWL próprio por split/chamada
            r0 = time.perf_counter()
            reasoner = run_owl_reasoner(onto, engine=self.reasoner_engine, infer_property_values=True, debug=0)
            self.n_reasoner_calls = int(getattr(self, "n_reasoner_calls", 0)) + 1
            reasoner_time = time.perf_counter() - r0
            gate = OntologyQualityGate()
            report = gate.evaluate(names, onto, reasoner_report=reasoner, require_reasoner=True)
        except Exception as exc:  # OWL inválida não derruba o benchmark
            return empty(f"OWL inválida: {type(exc).__name__}: {exc}", False)
        if not report.accepted:
            ctx = empty(f"enriquecimento rejeitado pelo quality gate: {report.issues}", True)
            ctx.reasoner_time = reasoner_time
            return ctx
        accepted = [m for m in report.matches if m.get("accepted")]
        processor = OntologyProcessor(onto, quality_gate=gate)
        processor.fit(pd.DataFrame(np.asarray(X_train, dtype=float), columns=names), accepted_matches=accepted, log=False)
        enriched = [str(n) for n in processor.output_features_]
        if len(enriched) <= n_orig:
            ctx = empty("a ontologia não gerou features derivadas não duplicadas", True)
            ctx.reasoner_time = reasoner_time
            return ctx

        def transform(Z):
            return processor.transform(pd.DataFrame(np.asarray(Z, dtype=float), columns=names)).to_numpy(dtype=float)

        graph = OntologySemanticGraph.from_ontology(onto, accepted_matches=accepted)
        n = len(enriched)
        groups: List[Optional[str]] = []
        entities: List[Optional[str]] = []
        for k, nm in enumerate(enriched):
            base = nm
            gl = [g for g in (graph.feature_groups.get(base) or []) if not str(g).startswith("<class") and g not in JUNK_GROUPS]
            if not gl and nm.startswith("onto_"):
                core = nm[5:].rsplit("_aggregate", 1)[0]
                gl = [core] if core in graph.nodes else []
            groups.append(gl[-1] if gl else None)
            entities.append(graph.entity_for_feature(base))
        rel = np.eye(n)
        for i in range(n):
            for j in range(i + 1, n):
                if entities[i] and entities[j]:
                    rel[i, j] = rel[j, i] = float(graph.relatedness(entities[i], entities[j]))
        depths = np.array([float(graph.get_feature_depth(nm)) if k < n_orig else 1.0 for k, nm in enumerate(enriched)])
        depths = np.clip(np.nan_to_num(depths, nan=0.0), 0.0, 1.0)
        weights = np.ones(n)
        weights[n_orig:] = self.onto_weight
        return SemanticContext("real", self.name, True, names, enriched, list(range(n_orig)), list(range(n_orig, n)),
                               transform, weights, groups, rel, depths, ontology_valid=True,
                               semantic_time=time.perf_counter() - t0 - reasoner_time, reasoner_time=reasoner_time,
                               info={"ontology_quality": report.status, "reasoner": reasoner.get("engine"),
                                     "ontology_hash": hashlib.sha256(Path(self.ontology_path).read_bytes()).hexdigest(),
                                     "n_ontology_graph_nodes": int(len(graph.nodes)),
                                     "mapped_feature_count": len(accepted), "unmapped_feature_count": max(0, n_orig - len(accepted)),
                                     "reasoning_applied": bool(reasoner), "reasoner_consistent": reasoner.get("consistent"),
                                     "feature_coverage": report.metrics.get("feature_coverage")})


class ShuffledSemanticProvider:
    """CONTROLO NEGATIVO: preserva dimensionalidade, nº de features derivadas, distribuição dos pesos/grupos/
    relatedness e protocolo, mas DESTRÓI a correspondência semântica real.

    * a semântica por feature (peso, grupo, depth, linhas/colunas da relatedness) é permutada;
    * as features derivadas são calculadas sobre colunas originais permutadas (entidade ↔ coluna errada)."""
    name = "owl_shuffled"

    def __init__(self, base, seed_offset: int = 7919):
        self.base = base
        self.seed_offset = int(seed_offset)

    def build(self, X_train, feature_names, seed: int = 0) -> SemanticContext:
        return self.from_context(self.base.build(X_train, feature_names, seed), seed)

    def from_context(self, ctx: SemanticContext, seed: int = 0) -> SemanticContext:
        """Permuta uma semântica JÁ construída (resultado idêntico a ``build``, sem recarregar a OWL nem repetir o reasoning)."""
        if not ctx.available:
            ctx.provider = self.name
            return ctx
        rng = np.random.default_rng(int(seed) + self.seed_offset)
        n_orig, n = len(ctx.orig_idx), ctx.n_features
        col_perm = rng.permutation(n_orig)
        perm = np.concatenate([rng.permutation(n_orig), n_orig + rng.permutation(n - n_orig)])
        real_transform = ctx.transform

        def shuffled_transform(Z):
            Z = np.asarray(Z, dtype=float)
            full = real_transform(Z[:, col_perm])
            full[:, :n_orig] = Z                      # as features originais mantêm-se (só a semântica é destruída)
            return full

        sh = SemanticContext(
            "shuffled", self.name, True, ctx.base_names, ctx.enriched_names, ctx.orig_idx, ctx.onto_idx,
            shuffled_transform, ctx.weights[perm], [ctx.groups[i] for i in perm], ctx.relatedness[np.ix_(perm, perm)],
            ctx.depths[perm], ontology_valid=ctx.ontology_valid, reason="controlo negativo: semântica permutada",
            semantic_time=ctx.semantic_time, reasoner_time=ctx.reasoner_time,
            info={**ctx.info, "shuffle_seed": int(seed) + self.seed_offset, "original_ontology_hash": ctx.info.get("ontology_hash"),
                  "counts_before": structure_counts(ctx), "n_ontology_graph_nodes": ctx.info.get("n_ontology_graph_nodes"),
                  "perm_hash": hashlib.sha256(perm.tobytes() + col_perm.tobytes()).hexdigest()[:16],
                  "real_structure_signature": ctx.structure_signature()})
        sh.info["counts_after"] = structure_counts(sh)
        sh.info["shuffled_structure_signature"] = sh.structure_signature()
        sh.info["shuffled_ontology_hash"] = hashlib.sha256("|".join(
            [str(sh.info["original_ontology_hash"]), str(sh.info["shuffle_seed"]), sh.info["perm_hash"],
             sh.info["shuffled_structure_signature"]]).encode()).hexdigest()
        return sh
