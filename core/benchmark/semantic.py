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


class ConventionalFeatureProvider:
    """CONTROLO de engenharia convencional de features (sem ontologia): mesma FORMA do enriquecimento OWL (colunas originais + agregados = média
    padronizada de um grupo de colunas), mas os grupos vêm dos DADOS DE TREINO (agrupamento hierárquico de ligação média sobre 1−|ρ_Spearman|), não de
    uma ontologia. Número de agregados ≤ ``n_target`` (o do OWL, para orçamento comparável) e ≤ p//2; só grupos com ≥2 colunas. Determinístico, sem RNG,
    agnóstico ao dataset. Estrutura semântica neutra (pesos 1, sem grupos, relatedness identidade, depths 0)."""
    name = "conventional_features"

    def __init__(self, n_target: int):
        self.n_target = int(n_target)

    def build(self, X_train, feature_names, seed: int = 0) -> SemanticContext:
        from scipy.cluster.hierarchy import fcluster, linkage
        from scipy.spatial.distance import squareform
        from scipy.stats import rankdata
        X = np.asarray(X_train, dtype=float)
        names = [str(n) for n in feature_names]
        p = X.shape[1]
        k = min(self.n_target, p // 2)
        sd = X.std(axis=0)
        ok = np.where(sd > 1e-12)[0]
        if k < 1 or len(ok) < 2:
            return SemanticContext("none", self.name, False, names, names, list(range(p)), [], reason="sem grupos convencionais possíveis")
        ranks = np.column_stack([rankdata(X[:, j]) for j in ok])
        corr = np.nan_to_num(np.corrcoef(ranks, rowvar=False), nan=0.0)
        dist = np.clip(1.0 - np.abs(corr), 0.0, None)
        np.fill_diagonal(dist, 0.0)
        labels = fcluster(linkage(squareform(dist, checks=False), method="average"), t=min(k, len(ok) - 1), criterion="maxclust")
        groups = [ok[np.where(labels == c)[0]] for c in sorted(set(labels))]
        groups = [g for g in groups if len(g) >= 2]
        if not groups:
            return SemanticContext("none", self.name, False, names, names, list(range(p)), [], reason="sem grupos convencionais com ≥2 colunas")
        centers = X.mean(axis=0)
        scales = np.where(sd > 1e-12, sd, 1.0)

        def transform(Z):
            Z = np.asarray(Z, dtype=float)
            derived = [((Z[:, g] - centers[g]) / scales[g]).mean(axis=1) for g in groups]
            return np.column_stack([Z] + derived)

        enriched = names + [f"conv_g{i}_mean" for i in range(len(groups))]
        n = len(enriched)
        return SemanticContext("conventional", self.name, True, names, enriched, list(range(p)), list(range(p, n)), transform,
                               np.ones(n), [None] * n, np.eye(n), np.zeros(n), ontology_valid=None,
                               reason="controlo de engenharia convencional (sem ontologia)",
                               info={"conventional_groups": [[names[j] for j in g] for g in groups], "n_target": self.n_target})


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

    def from_context(self, ctx: SemanticContext, seed: int = 0, mode: str = "both") -> SemanticContext:
        """Permuta uma semântica JÁ construída (resultado idêntico a ``build``, sem recarregar a OWL nem repetir o reasoning).

        ``mode`` (controlos P2, Doc. 23): ``both`` (histórico: colunas de origem E estrutura permutadas), ``structure`` (features VERDADEIRAS, estrutura
        — pesos/grupos/relatedness/profundidades — permutada) ou ``features`` (features calculadas sobre colunas permutadas, estrutura VERDADEIRA).
        Os sorteios do RNG são os mesmos nos três modos (``both`` inalterado)."""
        if mode not in ("both", "structure", "features"):
            raise ValueError("mode deve ser both|structure|features")
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

        use_perm_structure = mode in ("both", "structure")
        use_shuffled_transform = mode in ("both", "features")
        ident = np.arange(n)
        sperm = perm if use_perm_structure else ident
        sh = SemanticContext(
            "shuffled", self.name, True, ctx.base_names, ctx.enriched_names, ctx.orig_idx, ctx.onto_idx,
            shuffled_transform if use_shuffled_transform else real_transform, ctx.weights[sperm], [ctx.groups[i] for i in sperm],
            ctx.relatedness[np.ix_(sperm, sperm)], ctx.depths[sperm], ontology_valid=ctx.ontology_valid,
            reason="controlo negativo: semântica permutada" if mode == "both" else f"controlo parcial P2: {mode} permutado",
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


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# Controlos ontológicos do Doc. 29 (EXPERIMENTAIS; nunca usados por omissão). Ambos reutilizam o MESMO processador/quality gate/reasoner que a OWL real, por isso mantêm
# dimensão, receita de agregação, ESCALA (o processador é reajustado ao treino com as colunas já trocadas) e custo de pesquisa. Não alteram o núcleo do TREPAN.
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def column_groups(ctx: SemanticContext) -> List[Optional[str]]:
    """Rótulo de grupo de cada coluna ORIGINAL (None = sem mapeamento/geral)."""
    out = []
    for i in ctx.orig_idx:
        g = ctx.groups[i] if ctx.groups is not None and i < len(ctx.groups) else None
        out.append(None if (g is None or str(g).lower() in {"general", "none", ""}) else str(g))
    return out


def group_retention(groups: Sequence[Optional[str]], perm: np.ndarray) -> float:
    """Fração dos pares de colunas CO-AGRUPADAS na ontologia real que continuam co-agrupados depois de a coluna j passar a transportar a entidade ``perm[j]``."""
    g = list(groups)
    n = len(g)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if g[i] is not None and g[i] == g[j]]
    if not pairs:
        return 0.0
    return float(np.mean([g[perm[i]] is not None and g[perm[i]] == g[perm[j]] for i, j in pairs]))


def chance_retention(groups: Sequence[Optional[str]]) -> float:
    """Valor esperado de ``group_retention`` sob permutação uniforme: P(duas colunas distintas partilham o grupo) = Σ n_g(n_g−1) / (n(n−1))."""
    from collections import Counter
    n = len(groups)
    if n < 2:
        return 0.0
    c = Counter(x for x in groups if x is not None)
    return float(sum(v * (v - 1) for v in c.values()) / (n * (n - 1)))


def scale_strata(X_train, *, min_block: int = 8) -> np.ndarray:
    """Estrato de ESCALA de cada coluna (genérico, só treino): colunas ordenadas por log10(DP) e cortadas em ⌊p/min_block⌋ blocos contíguos de tamanho quase igual (≥ ``min_block``
    quando p ≥ ``min_block``; um só bloco se p < 2·``min_block``). Permutar só DENTRO de cada estrato evita razões/agregados absurdos entre colunas de magnitudes muito
    diferentes (artefacto de escala que tornaria o controlo artificialmente pior), sem usar nomes de datasets."""
    X = np.asarray(X_train, dtype=float)
    p = X.shape[1]
    k = max(1, p // int(min_block))
    order = np.argsort(np.log10(np.maximum(X.std(axis=0), 1e-12)), kind="stable")
    strata = np.zeros(p, dtype=int)
    for b, idx in enumerate(np.array_split(order, k)):
        strata[idx] = b
    return strata


def _stratified_permutation(rng, strata: Optional[np.ndarray], n: int) -> np.ndarray:
    if strata is None:
        return rng.permutation(n)
    perm = np.arange(n)
    for s in np.unique(strata):
        idx = np.where(strata == s)[0]
        perm[idx] = rng.permutation(idx)
    return perm


def select_destructive_permutation(groups: Sequence[Optional[str]], seed: int, *, strata: Optional[np.ndarray] = None, max_candidates: int = 5000) -> Tuple[np.ndarray, dict]:
    """Primeira permutação (fluxo de candidatas independente por semente: ``default_rng([seed, c])``, c = 0, 1, ...; opcionalmente só dentro de estratos de escala) cuja retenção de pares
    co-agrupados é ≤ à esperada ao acaso e que não fixa todas as colunas. Critério SÓ estrutural (sem rótulos nem resultados). Se não existir, devolve a de menor retenção
    e regista ``met_criterion=False``."""
    n = len(groups)
    thr = chance_retention(groups)
    best = None
    for c in range(max_candidates):
        p = _stratified_permutation(np.random.default_rng([int(seed), c]), strata, n)
        r = group_retention(groups, p)
        fixed = float(np.mean(p == np.arange(n)))
        if fixed < 1.0 and r <= thr + 1e-12:
            return p, {"seed_used": int(seed), "candidate_index": c, "candidates_tried": c + 1, "retention": r, "chance_retention": thr, "fixed_point_fraction": fixed, "met_criterion": True,
                       "scale_stratified": strata is not None}
        if best is None or r < best[1]:
            best = (p, r, c, fixed)
    return best[0], {"seed_used": int(seed), "candidate_index": best[2], "candidates_tried": max_candidates, "retention": best[1], "chance_retention": thr, "fixed_point_fraction": best[3], "met_criterion": False,
                     "scale_stratified": strata is not None}


class NamePermutedOwlProvider:
    """CONTROLO «ontologia permutada» (acoplado): a MESMA OWL, mas o quality gate/matching/processador recebem as colunas com os NOMES permutados (a coluna j passa a
    transportar a entidade da coluna ``perm[j]``). Destrói a correspondência coluna↔entidade (pertença a grupos, composição de cada derivada, relações); preserva o nº de
    entidades/grupos, a distribuição de tamanhos dos grupos, a dimensão e a receita das derivadas, pesos/profundidades/relatedness do grafo, o estado do quality gate (mesmo
    conjunto de nomes) e a ESCALA (processador reajustado ao treino já com colunas trocadas). A estrutura permanece COERENTE com as derivadas (como numa ontologia real aplicada às colunas erradas)."""
    name = "owl_name_permuted"

    def __init__(self, base: "OwlSemanticProvider", perm_seed: int):
        self.base, self.perm_seed = base, int(perm_seed)

    def build(self, X_train, feature_names, seed: int = 0, real_ctx: Optional[SemanticContext] = None) -> SemanticContext:
        names = [str(n) for n in feature_names]
        real = real_ctx if real_ctx is not None else self.base.build(X_train, names, seed)
        if not real.available:
            return real
        perm, meta = select_destructive_permutation(column_groups(real), self.perm_seed, strata=scale_strata(X_train))
        if not meta["met_criterion"]:                           # estratos de escala alinhados com os grupos não permitem destruir a correspondência: usa permutação livre (a porta de validade decide)
            perm, meta = select_destructive_permutation(column_groups(real), self.perm_seed)
            meta["scale_stratified"] = False
            meta["stratified_attempt_failed"] = True
        permuted_names = [names[j] for j in perm]
        ctx = self.base.build(np.asarray(X_train, dtype=float), permuted_names, seed)
        if not ctx.available:                                  # o conjunto de nomes é o mesmo ⇒ o gate não deve divergir; se divergir, regista-se (nunca se finge controlo)
            ctx.reason = "controlo permutado indisponível: " + ctx.reason
            return ctx
        ctx.provider, ctx.source = self.name, "shuffled"
        ctx.base_names = names
        ctx.enriched_names = names + list(ctx.enriched_names[len(names):])
        ctx.reason = "controlo negativo: ontologia permutada (nomes de colunas permutados; acoplado)"
        ctx.info = {**ctx.info, "control": "name_permuted", "perm_meta": meta, "perm_hash": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()[:16],
                    "real_structure_signature": real.structure_signature(), "original_ontology_hash": real.info.get("ontology_hash"),
                    "shuffled_structure_signature": ctx.structure_signature(), "shuffled_ontology_hash": hashlib.sha256(
                        "|".join([str(real.info.get("ontology_hash")), str(self.perm_seed), str(meta["candidate_index"]), ctx.structure_signature()]).encode()).hexdigest()}
        return ctx


class DecoupledRandomOwlProvider:
    """CONTROLO «grupos semânticos aleatórios» (desacoplado): composição das derivadas = a da OWL aplicada a nomes permutados por uma permutação A; rótulos de grupo,
    pesos, profundidades e relatedness = os da OWL real permutados por uma permutação B INDEPENDENTE (dentro das colunas originais e dentro das derivadas). Mantém tamanhos e nº
    de grupos, dimensão, receita e escala; destrói também o acoplamento entre rótulos e composição (o que o controlo acoplado preserva). Em TBoxes só de partição os dois controlos
    coincidem em lei; distinguem-se em ontologias com mais estrutura."""
    name = "owl_random_groups"

    def __init__(self, base: "OwlSemanticProvider", seed_a: int, seed_b: int):
        self.base, self.seed_a, self.seed_b = base, int(seed_a), int(seed_b)

    def build(self, X_train, feature_names, seed: int = 0, real_ctx: Optional[SemanticContext] = None) -> SemanticContext:
        names = [str(n) for n in feature_names]
        real = real_ctx if real_ctx is not None else self.base.build(X_train, names, seed)
        if not real.available:
            return real
        ctx = NamePermutedOwlProvider(self.base, self.seed_a).build(X_train, names, seed, real_ctx=real)
        if not ctx.available:
            return ctx
        n_o, n = len(real.orig_idx), real.n_features
        rng = np.random.default_rng(self.seed_b)
        perm = np.concatenate([rng.permutation(n_o), n_o + rng.permutation(n - n_o)])
        ctx.provider = self.name
        ctx.weights = real.weights[perm]
        ctx.groups = [real.groups[i] for i in perm]
        ctx.relatedness = real.relatedness[np.ix_(perm, perm)]
        ctx.depths = real.depths[perm]
        ctx.reason = "controlo negativo: grupos semânticos aleatórios (composição e rótulos independentes)"
        ctx.info = {**ctx.info, "control": "random_groups", "seed_a": self.seed_a, "seed_b": self.seed_b, "structure_perm_hash": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()[:16],
                    "shuffled_structure_signature": ctx.structure_signature()}
        return ctx


def control_validity(real: SemanticContext, ctx: SemanticContext, X_train, *, max_std_factor: float = 100.0) -> dict:
    """Porta de validade A PRIORI de um controlo (nunca usa resultados): mesma dimensão derivada que a OWL real, valores finitos e escala das derivadas não superior a
    ``max_std_factor`` × a da OWL real. Evita controlos artificialmente piores por artefactos de permutação (derivadas descartadas, razões com denominadores quase nulos)."""
    if not ctx.available:
        return {"valid": False, "reasons": ["indisponível: " + ctx.reason]}
    n_o = len(real.orig_idx)
    Er, Ec = real.enrich(X_train)[:, n_o:], ctx.enrich(X_train)[:, n_o:]
    reasons = []
    if Ec.shape[1] != Er.shape[1]:
        reasons.append(f"dimensão derivada {Ec.shape[1]} ≠ {Er.shape[1]}")
    if not np.isfinite(Ec).all():
        reasons.append("derivadas não finitas")
    pm = (ctx.info or {}).get("perm_meta")
    if pm is not None and not pm.get("met_criterion", True):
        reasons.append("permutação não cumpre o critério de destruição (retenção > acaso)")
    if Ec.shape[1] and Er.shape[1] and float(Ec.std(axis=0).max()) > max_std_factor * float(Er.std(axis=0).max()):
        reasons.append(f"escala {float(Ec.std(axis=0).max()):.3g} > {max_std_factor}× a da OWL real ({float(Er.std(axis=0).max()):.3g})")
    return {"valid": not reasons, "reasons": reasons, "n_derived": int(Ec.shape[1]), "n_derived_real": int(Er.shape[1]), "std_max": float(Ec.std(axis=0).max()) if Ec.size else 0.0,
            "std_max_real": float(Er.std(axis=0).max()) if Er.size else 0.0}


def build_valid_control(kind: str, base: "OwlSemanticProvider", real: SemanticContext, X_train, feature_names, seed: int, seed_a: int, seed_b: int, *, max_attempts: int = 4) -> Tuple[SemanticContext, dict]:
    """Constrói um controlo ``kind ∈ {"permuted","random_groups"}`` aplicando a porta de validade a sucessivas permutações (semente + 997·tentativa, pré-definidas). Se nenhuma
    for válida, recorre ao controlo PARCIAL «estrutura permutada» (features reais, estrutura permutada: dimensão e escala exatas, mas a representação conserva o conteúdo
    das derivadas verdadeiras) e regista ``variant='structure_only_fallback'`` — a limitação fica explícita e o controlo nunca é apresentado como o completo."""
    attempts = []
    for a in range(max_attempts):
        if kind == "permuted":
            ctx = NamePermutedOwlProvider(base, seed_a + 997 * a).build(X_train, feature_names, seed, real_ctx=real)
        else:
            ctx = DecoupledRandomOwlProvider(base, seed_a + 997 * a, seed_b + 997 * a).build(X_train, feature_names, seed, real_ctx=real)
        v = control_validity(real, ctx, X_train)
        attempts.append({"attempt": a, "valid": v["valid"], "reasons": v["reasons"]})
        if v["valid"]:
            ctx.info = {**ctx.info, "control_validity": v, "control_variant": "full", "attempts": attempts}
            return ctx, {"variant": "full", "attempts": attempts, "validity": v}
    fb = ShuffledSemanticProvider(None).from_context(base.build(X_train, feature_names, seed), seed_a, mode="structure")
    fb.info = {**fb.info, "control_variant": "structure_only_fallback", "attempts": attempts}
    return fb, {"variant": "structure_only_fallback", "attempts": attempts, "validity": control_validity(real, fb, X_train)}
