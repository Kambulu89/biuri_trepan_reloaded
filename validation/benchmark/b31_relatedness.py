"""B3.1 (Doc. 35): relatedness TAXONÓMICA sobre o registo de factos (``b31_facts``), só em ``validation/``. Não substitui nem altera ``OntologySemanticGraph.relatedness``.

Motivação medida na B3.0: a medida histórica devolve 0,712448 = 0,88·0,92·0,88 para qualquer par de propriedades de dados (caminho pelo nó genérico ``DatatypeProperty``).
Aqui um ancestral só conta se for INFORMATIVO: a sua extensão (atributos mapeados a que se aplica) é um subconjunto próprio do conjunto de atributos mapeados.

Definições (sem constantes ajustáveis):
* S(f)  = conceitos (e pseudo-conceitos de propriedade ``prop:q``/documentais ``doc:k``) aplicáveis ao atributo f, com fecho de subsunção, equivalência, sub-propriedade e
  subsunção inferida pelo reasoner (factos ``ATTRIBUTE_APPLIES_TO_CONCEPT`` disponíveis e não rejeitados).
* E(c)  = {f : c ∈ S(f)};   N = nº de atributos mapeados.   c é informativo ⇔ 0 < |E(c)| < N.
* IC(c) = −log(|E(c)|/N);   IC(f) = max_{c informativo ∈ S(f)} IC(c)   (conceito mais específico).
* lin(f,g) = 2·max_{c ∈ S(f)∩S(g) informativo} IC(c) / (IC(f)+IC(g))   ∈ [0,1];   lin(f,f)=1.
* wup(f,g) = 2·d(lcs)/(d(f)+d(g)) com profundidade ``d`` no DAG de conceitos informativos (raiz informativa = 1).
Saída tri-estado (mundo aberto): RELATED (score>0) | DISTINCT_CONCEPTS_DISJOINT (disjunção declarada entre conceitos de f e de g; evidência de conceitos distintos, NÃO de
incompatibilidade de combinação) | NO_RELATIONAL_EVIDENCE (ausência de evidência ≠ evidência de incompatibilidade; score 0)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from validation.benchmark.b31_facts import Epistemic, FactRegistry, Kind, PermittedUse, UseState

RELATED = "RELATED"
DISTINCT_DISJOINT = "DISTINCT_CONCEPTS_DISJOINT"
NO_EVIDENCE = "NO_RELATIONAL_EVIDENCE"


@dataclass
class RelatednessResult:
    score: float
    state: str
    common_concepts: Tuple[str, ...] = ()
    basis_fact_ids: Tuple[str, ...] = ()
    reason: str = ""


@dataclass
class ConceptIndex:
    features: List[str]
    concepts_of: Dict[str, Dict[str, Tuple[str, str]]]       # f -> {concept: (derivation, fact_id)}
    extension: Dict[str, set]
    ic: Dict[str, float]
    informative: set
    parents: Dict[str, set]                                    # só entre conceitos informativos
    disjoint_pairs: List[Tuple[str, str, str]] = field(default_factory=list)   # (c1, c2, fact_id)
    n_mapped: int = 0


def _usable(f) -> bool:
    return f.use_state != UseState.REJECTED.value and f.epistemic in (Epistemic.LOGICALLY_ENTAILED.value, Epistemic.DOMAIN_DOCUMENTED.value) and f.permitted_use != PermittedUse.NONE.value


def build_index(reg: FactRegistry, features: Sequence[str]) -> ConceptIndex:
    feats = [str(f) for f in features]
    concepts_of: Dict[str, Dict[str, Tuple[str, str]]] = {f: {} for f in feats}
    mapped = {f for f in feats if reg.feature_entity.get(f)}
    for fa in reg.by_kind(Kind.ATTRIBUTE_APPLIES_TO_CONCEPT):
        if not _usable(fa) or fa.attributes[0] not in concepts_of:
            continue
        concepts_of[fa.attributes[0]].setdefault(fa.entities[1], (fa.derivation, fa.id))
    # pseudo-conceitos de propriedade (sub-propriedade / equivalência), com fecho
    pe = {}
    for fp in reg.by_kind(Kind.PROPERTY_SUBSUMPTION, Kind.PROPERTY_EQUIVALENCE):
        if fp.use_state == UseState.REJECTED.value:
            continue
        pe.setdefault(fp.kind, []).append(fp)
    ent2f: Dict[str, List[str]] = {}
    for f in feats:
        e = reg.feature_entity.get(f)
        if e:
            ent2f.setdefault(e, []).append(f)
    up: Dict[str, Dict[str, str]] = {}                       # propriedade -> {super/equivalente: fact_id}
    for fp in pe.get(Kind.PROPERTY_SUBSUMPTION.value, []):
        up.setdefault(fp.entities[0], {})[fp.entities[1]] = fp.id
    for fp in pe.get(Kind.PROPERTY_EQUIVALENCE.value, []):
        a, b = fp.entities
        up.setdefault(a, {})[b] = fp.id
        up.setdefault(b, {})[a] = fp.id
    changed = True
    while changed:
        changed = False
        for p, sups in list(up.items()):
            for q, fid in list(sups.items()):
                for r, fid2 in up.get(q, {}).items():
                    if r != p and r not in sups:
                        sups[r] = fid2; changed = True
    canon: Dict[str, str] = {}
    for fp in pe.get(Kind.PROPERTY_EQUIVALENCE.value, []):
        a, b = fp.entities
        ra, rb = canon.get(a, a), canon.get(b, b)
        r = min(ra, rb)
        for k, v in list(canon.items()):
            if v in (ra, rb):
                canon[k] = r
        canon[a] = canon[b] = r
    cn = lambda x: canon.get(x, x)
    for f in feats:
        e = reg.feature_entity.get(f)
        if not e:
            continue
        concepts_of[f].setdefault(f"prop:{cn(e)}", ("ASSERTED", f"prop-self:{cn(e)}"))
        for q, fid in up.get(e, {}).items():
            concepts_of[f].setdefault(f"prop:{cn(q)}", ("DERIVED_CLOSURE", fid))
    # relações documentadas: pseudo-conceito por facto
    for fd in reg.by_kind(Kind.SAME_MEASUREMENT_FAMILY):
        if not _usable(fd):
            continue
        for a in fd.attributes:
            if a in concepts_of:
                concepts_of[a].setdefault(f"doc:{fd.id}", ("DOCUMENT", fd.id))
    n = max(len(mapped), 1)
    ext: Dict[str, set] = {}
    for f in feats:
        for c in concepts_of[f]:
            ext.setdefault(c, set()).add(f)
    # propriedade que NÃO está noutro atributo (prop:self só do próprio) tem |E|=1: informativa mas sem partilha
    informative = {c for c, e in ext.items() if 0 < len(e) < n or (len(e) == n == 1)}
    ic = {c: -math.log(len(ext[c]) / n) for c in informative}
    # pais entre conceitos informativos (taxonomia): c é pai de d se E(d) ⊂ E(c) estrito e não existe intermédio (aprox. por inclusão de extensão)
    parents: Dict[str, set] = {c: set() for c in informative}
    for c in informative:
        for d in informative:
            if c != d and ext[c] < ext[d]:
                parents[c].add(d)
    disj = []
    for fd in reg.by_kind(Kind.CONCEPT_DISJOINTNESS):
        if fd.use_state != UseState.REJECTED.value:
            disj.append((fd.entities[0], fd.entities[1], fd.id))
    return ConceptIndex(feats, concepts_of, ext, ic, informative, parents, disj, len(mapped))


def _ic_attr(ix: ConceptIndex, f: str) -> float:
    return max((ix.ic[c] for c in ix.concepts_of.get(f, {}) if c in ix.informative), default=0.0)


def _depth(ix: ConceptIndex, c: str, memo: Dict[str, int]) -> int:
    if c in memo:
        return memo[c]
    ps = ix.parents.get(c, set())
    memo[c] = 1 if not ps else 1 + min(_depth(ix, p, memo) for p in ps)
    return memo[c]


def pair_relatedness(ix: ConceptIndex, f: str, g: str, measure: str = "lin") -> RelatednessResult:
    if f == g:
        return RelatednessResult(1.0, RELATED, reason="identidade")
    sf, sg = ix.concepts_of.get(f, {}), ix.concepts_of.get(g, {})
    if not sf or not sg:
        return RelatednessResult(0.0, NO_EVIDENCE, reason="atributo sem conceitos aplicáveis (matching/ontologia)")
    common = [c for c in sf if c in sg and c in ix.informative]
    if common:
        if measure == "lin":
            best = max(common, key=lambda c: (ix.ic[c], c))
            denom = _ic_attr(ix, f) + _ic_attr(ix, g)
            score = 1.0 if denom <= 1e-12 else min(1.0, max(0.0, 2.0 * ix.ic[best] / denom))
        else:
            memo: Dict[str, int] = {}
            best = max(common, key=lambda c: (_depth(ix, c, memo), c))
            df = max((_depth(ix, c, memo) for c in sf if c in ix.informative), default=1)
            dg = max((_depth(ix, c, memo) for c in sg if c in ix.informative), default=1)
            score = min(1.0, 2.0 * _depth(ix, best, memo) / (df + dg))
        fids = tuple(sorted({sf[c][1] for c in common if c in sf} | {sg[c][1] for c in common if c in sg}))
        return RelatednessResult(float(score), RELATED, tuple(sorted(common)), fids)
    # sem conceito informativo comum: evidência de conceitos distintos (disjunção declarada) ou ausência de evidência
    names_f, names_g = set(sf), set(sg)
    for a, b, fid in ix.disjoint_pairs:
        if (a in names_f and b in names_g) or (b in names_f and a in names_g):
            return RelatednessResult(0.0, DISTINCT_DISJOINT, (), (fid,), "disjunção declarada entre conceitos dos atributos")
    return RelatednessResult(0.0, NO_EVIDENCE, (), (), "sem conceito informativo comum")


def relatedness_matrix(reg: FactRegistry, features: Sequence[str], measure: str = "lin") -> Tuple[np.ndarray, List[List[str]]]:
    ix = build_index(reg, features)
    n = len(ix.features)
    M = np.eye(n)
    S = [[RELATED if i == j else NO_EVIDENCE for j in range(n)] for i in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            r = pair_relatedness(ix, ix.features[i], ix.features[j], measure)
            M[i, j] = M[j, i] = r.score
            S[i][j] = S[j][i] = r.state
    return M, S


@dataclass
class SemanticUnit:
    concept: str
    features: Tuple[str, ...]
    ic: float
    derivations: Tuple[str, ...]
    fact_ids: Tuple[str, ...]
    mean_matching_confidence: float


def semantic_units(reg: FactRegistry, features: Sequence[str], *, min_size: int = 2) -> List[SemanticUnit]:
    """Unidades semânticas = extensões de conceitos INFORMATIVOS com ≥ ``min_size`` atributos (conceitos genéricos, que cobrem tudo, não geram unidade). Extensões idênticas
    colapsam numa só (a mais específica por IC, depois nome). Ordem determinística: IC decrescente, depois tamanho, depois nome."""
    ix = build_index(reg, features)
    seen: Dict[Tuple[str, ...], SemanticUnit] = {}
    for c in sorted(ix.informative):
        if c.startswith("prop:") and len(ix.extension[c]) < min_size:
            continue
        ext = tuple(sorted(ix.extension[c]))
        if len(ext) < min_size:
            continue
        ders = tuple(sorted({ix.concepts_of[f][c][0] for f in ext}))
        fids = tuple(sorted({ix.concepts_of[f][c][1] for f in ext}))
        conf = float(np.mean([reg.match_confidence.get(f, 0.0) for f in ext]))
        u = SemanticUnit(c, ext, ix.ic[c], ders, fids, conf)
        old = seen.get(ext)
        if old is None or (u.ic, not u.concept.startswith('prop:'), u.concept) > (old.ic, not old.concept.startswith('prop:'), old.concept):
            seen[ext] = u
    return sorted(seen.values(), key=lambda u: (-u.ic, -len(u.features), u.concept))


def legacy_matrix(owl_path: str, features: Sequence[str]) -> np.ndarray:
    """Relatedness histórica (``OntologySemanticGraph.relatedness``) sobre os mesmos atributos, para comparação (núcleo intocado)."""
    from core.ontology_quality import OntologyQualityGate
    from core.ontology_semantic_graph import OntologySemanticGraph
    from core.owl_runtime import load_ontology_isolated
    onto = load_ontology_isolated(str(owl_path))
    dec = [d.__dict__ for d in OntologyQualityGate().match_features(list(features), onto) if d.accepted]
    g = OntologySemanticGraph.from_ontology(onto, accepted_matches=dec)
    n = len(features)
    M = np.eye(n)
    ents = [g.entity_for_feature(f) for f in features]
    for i in range(n):
        for j in range(i + 1, n):
            M[i, j] = M[j, i] = g.relatedness(ents[i], ents[j]) if ents[i] and ents[j] else 0.0
    return M
