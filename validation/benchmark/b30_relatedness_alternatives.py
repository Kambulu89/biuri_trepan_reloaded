"""B3.0 (Doc. 32): medidas de relatedness ALTERNATIVAS, só para diagnóstico (NÃO substituem ``OntologySemanticGraph.relatedness``; núcleo intocado).

Motivação medida: a relatedness atual devolve 0.7124 = 0.88·0.92·0.88 (dois saltos «subsumption», caminho máx-produto) para TODOS os pares de propriedades de dados,
porque cada propriedade tem uma aresta «subsumption» para o nó genérico ``DatatypeProperty`` (nó de topo W3C, ``builtin_nodes``) que o caminho atravessa.
O caminho via domínio (0.68·0.92·0.68 ≈ 0.425) nunca vence. Resultado: a medida não distingue pares do mesmo conceito de pares de conceitos distintos.

Propostas (matematicamente justificadas; só usam a taxonomia declarada — is-a e domínio — excluindo nós genéricos/de topo):
* Wu–Palmer:  wup(a,b) = 2·d(lca)/(d(a)+d(b)),  d = profundidade (raiz=1). Em [0,1], simétrica, reflexiva (wup(a,a)=1), monótona na especificidade do ancestral comum.
* Lin (1998): lin(a,b) = 2·IC(lca)/(IC(a)+IC(b)), IC(c) = −log p(c), p(c) = fração das features mapeadas cujo conceito é descendente de c. Usa SÓ a ontologia + o mapeamento
  (sem rótulos nem dados); um conceito que subsume tudo tem IC=0 ⇒ relação 0 (não há relação informativa).
Relatedness entre features = máximo sobre os pares de conceitos (domínio/superclasses) das duas features."""
from __future__ import annotations
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Set


@dataclass
class Taxonomy:
    parents: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))          # conceito → superconceitos (não genéricos)
    feature_concepts: Dict[str, Set[str]] = field(default_factory=dict)                        # feature → conceitos (domínio / superclasse direta)

    def closure(self, c: str) -> Set[str]:
        seen, stack = set(), [c]
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            stack.extend(self.parents.get(x, ()))
        return seen

    def depth(self, c: str, _visiting: Optional[Set[str]] = None) -> int:
        """Profundidade (raiz = 1) = 1 + mínimo das profundidades dos superconceitos; ciclos contam como raiz (igual a ``_entity_depth`` do núcleo, +1)."""
        v = set() if _visiting is None else _visiting
        if c in v:
            return 1
        ups = [p for p in self.parents.get(c, ()) if p != c]
        if not ups:
            return 1
        v.add(c)
        try:
            return 1 + min(self.depth(p, v) for p in ups)
        finally:
            v.discard(c)


def taxonomy_from_graph(graph, feature_to_entity: Dict[str, str]) -> Taxonomy:
    """Extrai a taxonomia de um ``OntologySemanticGraph`` JÁ construído (sem alterá-lo), excluindo ``builtin_nodes`` (Thing, DatatypeProperty, …)."""
    t = Taxonomy()
    builtin = set(graph.builtin_nodes) | {"Thing", "DatatypeProperty", "ObjectProperty", "Property"}
    for child, ups in graph.parents.items():
        for u in ups:
            if child not in builtin and u not in builtin:
                t.parents[child].add(u)
    for f, ent in feature_to_entity.items():
        concepts = {c for c in (graph.domains.get(ent) or set()) if c not in builtin}
        concepts |= {c for c in (graph.parents.get(ent) or set()) if c not in builtin}
        t.feature_concepts[f] = concepts
    return t


def wu_palmer(t: Taxonomy, a: str, b: str) -> float:
    common = t.closure(a) & t.closure(b)
    if not common:
        return 0.0
    d = max(t.depth(x) for x in common)
    return float(2.0 * d / (t.depth(a) + t.depth(b)))


def information_content(t: Taxonomy, features: Optional[Iterable[str]] = None) -> Dict[str, float]:
    feats = list(features if features is not None else t.feature_concepts)
    n = max(len(feats), 1)
    count: Dict[str, int] = defaultdict(int)
    for f in feats:
        seen: Set[str] = set()
        for c in t.feature_concepts.get(f, ()):
            seen |= t.closure(c)
        for c in seen:
            count[c] += 1
    return {c: -math.log(k / n) for c, k in count.items() if k > 0}


def lin(t: Taxonomy, a: str, b: str, ic: Dict[str, float]) -> float:
    common = (t.closure(a) & t.closure(b))
    if not common:
        return 0.0
    ic_l = max(ic.get(x, 0.0) for x in common)
    denom = ic.get(a, 0.0) + ic.get(b, 0.0)
    if denom <= 1e-12:
        return 1.0 if a == b else 0.0
    return float(min(1.0, max(0.0, 2.0 * ic_l / denom)))


def feature_relatedness(t: Taxonomy, f: str, g: str, measure: str = "wu_palmer", ic: Optional[Dict[str, float]] = None) -> Optional[float]:
    cf, cg = t.feature_concepts.get(f) or set(), t.feature_concepts.get(g) or set()
    if not cf or not cg:
        return None
    if measure == "lin":
        ic = ic if ic is not None else information_content(t)
        return max(lin(t, a, b, ic) for a in cf for b in cg)
    return max(wu_palmer(t, a, b) for a in cf for b in cg)


def relatedness_matrix(t: Taxonomy, features: List[str], measure: str = "wu_palmer"):
    import numpy as np
    ic = information_content(t, features) if measure == "lin" else None
    n = len(features)
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            v = feature_relatedness(t, features[i], features[j], measure, ic)
            M[i, j] = M[j, i] = 0.0 if v is None else v
    return M
