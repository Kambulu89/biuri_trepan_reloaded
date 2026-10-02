"""Grafo semântico OWL agnóstico ao dataset para o TREPAN Reloaded V9.2.

Extrai relações estruturais sem depender de nomes de datasets. O grafo é usado
apenas como conhecimento de domínio; não lê rótulos, splits nem resultados.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
import heapq
import logging
import re
import numpy as np
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

logger = logging.getLogger(__name__)


def _name(obj: Any) -> str:
    return str(getattr(obj, "name", None) or getattr(obj, "iri", None) or obj)


def normalize_ontology_name(value: Any) -> str:
    """Chave tolerante para casar colunas do dataset com conceitos OWL/RDF.

    Ignora prefixos de URI (``http://x.org/onto#``, ``.../onto/``), prefixos
    CURIE (``ex:``), maiúsculas/minúsculas, sublinhados, hífens, espaços e
    pontos. ``"http://x.org/o#Mean_Radius"``, ``"ex:meanRadius"`` e
    ``"MEAN_RADIUS"`` produzem todos ``"meanradius"``.
    """
    text = str(value or "").strip()
    if not text:
        return ""
    for sep in ("#", "/"):
        if sep in text:
            text = text.rsplit(sep, 1)[-1]
    if ":" in text:
        text = text.rsplit(":", 1)[-1]
    return re.sub(r"[\s_\-.]+", "", text).casefold()


def _is_builtin_entity(obj: Any) -> bool:
    """Entidades OWL/RDF(S)/XSD de topo não contam para a profundidade."""
    iri = str(getattr(obj, "iri", "") or "")
    return iri.startswith("http://www.w3.org/")


def _numeric_bound(value: Any) -> Optional[float]:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _has_integer_range(entity: Any) -> bool:
    """``rdfs:range`` xsd:integer (ou faceta sobre inteiro) numa data property."""
    for target in _iter_attr(entity, "range"):
        base = getattr(target, "base_datatype", target)
        if base is int or "integer" in str(getattr(base, "iri", "") or base).lower():
            return True
    return False


def _explicit_entity_bounds(entity: Any) -> Tuple[Optional[float], Optional[float]]:
    """Limites numéricos declarados na OWL (xsd:min/maxInclusive/Exclusive).

    Considera facetas directamente na entidade e em ``ConstrainedDatatype``
    usados como ``rdfs:range`` de propriedades de dados. Limites exclusivos são
    tratados como inclusivos: o filtro só existe para eliminar valores
    conceptualmente impossíveis, não para discriminar a fronteira exacta.
    """
    low: Optional[float] = None
    high: Optional[float] = None
    sources = [entity] + _iter_attr(entity, "range")
    for source in sources:
        for attr in ("min_inclusive", "minInclusive", "min_exclusive", "minExclusive"):
            bound = _numeric_bound(getattr(source, attr, None))
            if bound is not None:
                low = bound if low is None else max(low, bound)
        for attr in ("max_inclusive", "maxInclusive", "max_exclusive", "maxExclusive"):
            bound = _numeric_bound(getattr(source, attr, None))
            if bound is not None:
                high = bound if high is None else min(high, bound)
    return low, high


RELATION_WEIGHTS = {
    "equivalent": 1.00,
    "subsumption": 0.88,
    "inferred_subsumption": 0.78,
    "inverse": 0.82,
    "domain": 0.68,
    "range": 0.68,
}

def _iter_attr(obj: Any, attr: str) -> List[Any]:
    try:
        value = getattr(obj, attr, []) or []
        if isinstance(value, (str, bytes)):
            return [value]
        return list(value)
    except Exception:
        return []


@dataclass
class OntologySemanticGraph:
    nodes: Set[str] = field(default_factory=set)
    edges: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    edge_types: Dict[Tuple[str, str], Set[str]] = field(default_factory=lambda: defaultdict(set))
    labels: Dict[str, List[str]] = field(default_factory=dict)
    feature_to_entity: Dict[str, str] = field(default_factory=dict)
    feature_groups: Dict[str, List[str]] = field(default_factory=dict)
    entity_kinds: Dict[str, str] = field(default_factory=dict)
    # Hierarquia dirigida (filho -> pais) para medir a profundidade ontológica.
    parents: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    domains: Dict[str, Set[str]] = field(default_factory=lambda: defaultdict(set))
    entity_bounds: Dict[str, Tuple[Optional[float], Optional[float]]] = field(default_factory=dict)
    integer_entities: Set[str] = field(default_factory=set)
    source_path: Optional[str] = None
    load_error: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """A OWL só está activa se carregou e tem pelo menos um conceito."""
        return bool(self.load_error is None and self.nodes)

    @classmethod
    def load(
        cls,
        owl_path: str,
        *,
        accepted_matches: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "OntologySemanticGraph":
        """Carrega uma OWL/RDF do disco sem nunca interromper o pipeline.

        Em caso de falha devolve um grafo vazio com ``is_active=False``; o
        Reloaded trata esse estado exactamente como o TREPAN Original.
        """
        try:
            # Import tardio: o extractor importa este módulo.
            from core.trepan_reloaded_extractor import TrepanReloadedExtractor

            ontology = TrepanReloadedExtractor.load_ontology_file(owl_path)
            graph = cls.from_ontology(ontology, accepted_matches=accepted_matches)
            graph.source_path = str(owl_path)
            if not graph.nodes:
                raise ValueError("ontologia carregada sem classes nem propriedades")
            logger.info("Ontologia '%s' carregada e ativada com sucesso.", owl_path)
            return graph
        except Exception as exc:  # noqa: BLE001 - qualquer falha de parsing/IO cai no fallback
            logger.warning(
                "Falha ao carregar a ontologia '%s': %s. Fallback para Trepan Original ativo.",
                owl_path, exc,
            )
            graph = cls()
            graph.source_path = str(owl_path)
            graph.load_error = f"{type(exc).__name__}: {exc}"
            return graph

    @classmethod
    def from_ontology(
        cls,
        ontology,
        *,
        accepted_matches: Optional[Sequence[Mapping[str, Any]]] = None,
    ) -> "OntologySemanticGraph":
        graph = cls()
        if ontology is None:
            return graph
        entities: List[Any] = []
        entity_kind_by_id: Dict[int, str] = {}
        for method, kind in (
            ("classes", "class"),
            ("data_properties", "datatype_property"),
            ("object_properties", "object_property"),
            ("annotation_properties", "annotation_property"),
        ):
            try:
                values = list(getattr(ontology, method)())
            except Exception:
                values = []
            entities.extend(values)
            for entity in values:
                entity_kind_by_id[id(entity)] = kind
        seen = {}
        for entity in entities:
            n = _name(entity)
            if not n:
                continue
            graph.nodes.add(n); seen[n] = entity
            graph.entity_kinds[n] = entity_kind_by_id.get(id(entity), "unknown")
            labels = []
            for attr in ("label", "prefLabel", "altLabel"):
                labels.extend(str(v) for v in _iter_attr(entity, attr))
            graph.labels[n] = list(dict.fromkeys(labels))
            if _has_integer_range(entity):
                graph.integer_entities.add(n)
            low, high = _explicit_entity_bounds(entity)
            if low is not None or high is not None:
                graph.entity_bounds[n] = (low, high)

        relation_attrs = {
            "is_a": "subsumption",
            "equivalent_to": "equivalent",
            "domain": "domain",
            "range": "range",
            "inverse_property": "inverse",
        }
        for src_name, entity in seen.items():
            for attr, relation in relation_attrs.items():
                for target in _iter_attr(entity, attr):
                    tgt = _name(target)
                    if not tgt or tgt == src_name:
                        continue
                    if not _is_builtin_entity(target):
                        if relation == "subsumption":
                            graph.parents[src_name].add(tgt)
                        elif relation == "domain":
                            graph.domains[src_name].add(tgt)
                    graph.nodes.add(tgt)
                    graph.edges[src_name].add(tgt)
                    graph.edges[tgt].add(src_name)
                    graph.edge_types[(src_name, tgt)].add(relation)
                    graph.edge_types[(tgt, src_name)].add(relation)

            # Se o reasoner já materializou a hierarquia no objecto OWL,
            # ``ancestors()`` permite aproveitar essa inferência sem depender
            # do nome do dataset ou de axiomas específicos.
            try:
                ancestors = list(entity.ancestors()) if callable(getattr(entity, "ancestors", None)) else []
            except Exception:
                ancestors = []
            for target in ancestors[:64]:
                tgt = _name(target)
                if not tgt or tgt == src_name or tgt.lower().endswith("thing"):
                    continue
                graph.nodes.add(tgt)
                graph.edges[src_name].add(tgt)
                graph.edges[tgt].add(src_name)
                graph.edge_types[(src_name, tgt)].add("inferred_subsumption")
                graph.edge_types[(tgt, src_name)].add("inferred_subsumption")

        for item in accepted_matches or []:
            if not item.get("accepted", True):
                continue
            feature = str(item.get("feature", "")).strip()
            entity = str(item.get("entity_name", "")).strip()
            if feature and entity:
                # O quality gate pode devolver IRIs completos ou grafias
                # diferentes; ancora sempre no nome do nó presente no grafo.
                graph.feature_to_entity[feature] = graph.resolve_entity(entity) or entity

        graph._derive_feature_groups()
        return graph

    def _normalized_index(self, names: Iterable[str]) -> Dict[str, str]:
        index: Dict[str, str] = {}
        for name in sorted(map(str, names)):
            index.setdefault(normalize_ontology_name(name), name)
        return index

    def resolve_entity(self, name: Any) -> Optional[str]:
        """Nome do nó OWL correspondente, tolerando URI/caixa/sublinhados."""
        text = str(name or "").strip()
        if not text:
            return None
        if text in self.nodes:
            return text
        key = normalize_ontology_name(text)
        if not key:
            return None
        hit = self._normalized_index(self.nodes).get(key)
        if hit is not None:
            return hit
        for node, labels in sorted(self.labels.items()):
            if any(normalize_ontology_name(label) == key for label in labels):
                return node
        return None

    def entity_for_feature(self, feature: Any) -> Optional[str]:
        """Entidade OWL mapeada para uma coluna, com matching tolerante."""
        text = str(feature or "").strip()
        if not text:
            return None
        if text in self.feature_to_entity:
            return self.feature_to_entity[text]
        key = normalize_ontology_name(text)
        mapped = self._normalized_index(self.feature_to_entity).get(key)
        if mapped is not None:
            return self.feature_to_entity[mapped]
        # Sem matching aceite: só uma correspondência directa a um nó conta.
        return self.resolve_entity(text)

    def _entity_depth(self, entity: str, _visiting: Optional[Set[str]] = None) -> int:
        """Profundidade na hierarquia OWL (subsunção; domínio para propriedades).

        Uma propriedade de dados sem super-propriedade herda a profundidade da
        classe do seu ``rdfs:domain`` + 1, para que propriedades de conceitos
        mais específicos sejam também mais profundas. Ciclos contam como raiz.
        """
        visiting = set() if _visiting is None else _visiting
        if entity in visiting:
            return 0
        visiting.add(entity)
        try:
            ups = self.parents.get(entity) or self.domains.get(entity) or set()
            depths = [self._entity_depth(p, visiting) + 1 for p in sorted(ups)]
            return min(depths) if depths else 0
        finally:
            visiting.discard(entity)

    def get_feature_depth(self, feature: Any) -> float:
        """OntoDepth(A) em [0, 1], relativo ao nó mais profundo do grafo.

        Devolve 0.0 para features não mapeadas ou quando a OWL está inactiva,
        de modo que o termo α·OntoDepth desaparece no fallback Original.
        """
        if not self.is_active:
            return 0.0
        entity = self.entity_for_feature(feature)
        if entity is None:
            return 0.0
        cache = getattr(self, "_depth_cache", None)
        if cache is None:
            cache = {node: self._entity_depth(node) for node in self.nodes}
            self._depth_cache = cache
        max_depth = max(cache.values()) if cache else 0
        if max_depth <= 0:
            return 0.0
        return float(cache.get(entity, 0)) / float(max_depth)

    def feature_bounds(self, feature: Any) -> Tuple[Optional[float], Optional[float]]:
        entity = self.entity_for_feature(feature)
        if entity is None:
            return (None, None)
        return self.entity_bounds.get(entity, (None, None))

    def sample_validity_mask(self, X, feature_names: Sequence[str]) -> np.ndarray:
        """Linhas que respeitam os limites numéricos declarados na OWL.

        Apenas facetas explícitas (xsd:minInclusive, ...) são usadas; quantis
        de treino nunca são tratados como restrições ontológicas.
        """
        X = np.asarray(X, dtype=float)
        valid = np.ones(len(X), dtype=bool)
        if X.ndim != 2 or not self.is_active:
            return valid
        for j, name in enumerate(list(feature_names)[: X.shape[1]]):
            low, high = self.feature_bounds(name)
            column = X[:, j]
            if low is not None:
                valid &= ~(column < low)
            if high is not None:
                valid &= ~(column > high)
        return valid

    def is_integer_feature(self, feature: Any) -> bool:
        entity = self.entity_for_feature(feature)
        return bool(entity is not None and entity in self.integer_entities)

    def domain_constraints(
        self,
        feature_names: Sequence[str],
        reference_X=None,
        *,
        min_reference_coverage: float = 0.99,
    ) -> "OntologyDomainConstraints":
        """Restrições OWL aplicáveis à matriz efectivamente usada pela árvore.

        Os limites da OWL estão em unidades do domínio; se a matriz estiver
        escalada/codificada, aplicá-los destruiria a feature. Uma restrição só é
        usada quando pelo menos ``min_reference_coverage`` das linhas de treino
        já a cumprem (prova de que a coluna está nas unidades da OWL).
        """
        names = list(map(str, feature_names))
        ref = None if reference_X is None else np.asarray(reference_X, dtype=float)
        specs: Dict[int, Dict[str, Any]] = {}
        skipped: List[str] = []
        if self.is_active:
            for j, name in enumerate(names):
                low, high = self.feature_bounds(name)
                integer = self.is_integer_feature(name)
                if low is None and high is None and not integer:
                    continue
                if ref is not None and ref.ndim == 2 and j < ref.shape[1] and len(ref):
                    col = ref[:, j]
                    col = col[np.isfinite(col)]
                    ok = np.ones(len(col), dtype=bool)
                    if low is not None:
                        ok &= col >= low
                    if high is not None:
                        ok &= col <= high
                    bounds_fit = bool(len(col) == 0 or ok.mean() >= min_reference_coverage)
                    int_fit = bool(
                        len(col) == 0
                        or np.mean(np.isclose(col, np.round(col), atol=1e-9)) >= min_reference_coverage
                    )
                    if not bounds_fit:
                        low = high = None
                    integer = integer and int_fit
                    if low is None and high is None and not integer:
                        skipped.append(name)
                        continue
                specs[j] = {"feature": name, "low": low, "high": high, "integer": bool(integer)}
        return OntologyDomainConstraints(specs, len(names), skipped)


    def _derive_feature_groups(self) -> None:
        groups: Dict[str, List[str]] = {}
        for feature, entity in self.feature_to_entity.items():
            candidates = []
            for neighbor in sorted(self.edges.get(entity, set())):
                rels = self.edge_types.get((entity, neighbor), set())
                if rels & {"subsumption", "domain", "range", "equivalent"}:
                    candidates.append(neighbor)
            groups[feature] = candidates or [entity]
        self.feature_groups = groups

    def neighbors(self, entity: str) -> Set[str]:
        return set(self.edges.get(str(entity), set()))

    def shortest_path_length(self, left: str, right: str, max_depth: int = 6) -> Optional[int]:
        left, right = str(left), str(right)
        if left == right:
            return 0
        if left not in self.nodes or right not in self.nodes:
            return None
        queue = deque([(left, 0)]); seen = {left}
        while queue:
            node, depth = queue.popleft()
            if depth >= max_depth:
                continue
            for nxt in self.edges.get(node, set()):
                if nxt == right:
                    return depth + 1
                if nxt not in seen:
                    seen.add(nxt); queue.append((nxt, depth + 1))
        return None

    def _edge_strength(self, left: str, right: str) -> float:
        rels = self.edge_types.get((str(left), str(right)), set())
        if not rels:
            return 0.50 if str(right) in self.edges.get(str(left), set()) else 0.0
        return float(max(RELATION_WEIGHTS.get(r, 0.50) for r in rels))

    def relatedness(self, left: str, right: str, max_depth: int = 6) -> float:
        """Relatedness 0..1 com pesos por tipo de relação OWL.

        Relações de equivalência/subsunção valem mais que domain/range. Caminhos
        mais longos sofrem decaimento, evitando tratar qualquer ligação no grafo
        como semanticamente equivalente. Não usa rótulos nem métricas do dataset.
        """
        if not left or not right:
            return 0.0
        left, right = str(left), str(right)
        if left == right:
            return 1.0
        if left not in self.nodes or right not in self.nodes:
            return 0.0
        # max-product path (fila de prioridade), com decaimento por salto.
        heap=[(-1.0, 0, left)]
        best={left:1.0}
        while heap:
            neg, depth, node = heapq.heappop(heap)
            strength = -neg
            if node == right:
                return float(np.clip(strength, 0.0, 1.0))
            if depth >= int(max_depth):
                continue
            for nxt in self.edges.get(node, set()):
                edge = self._edge_strength(node, nxt)
                candidate = strength * edge * (1.0 if depth == 0 else 0.92)
                if candidate > best.get(nxt, 0.0) + 1e-12:
                    best[nxt]=candidate
                    heapq.heappush(heap, (-candidate, depth+1, nxt))
        return 0.0

    def feature_relatedness_matrix(self, feature_names: Sequence[str]) -> List[List[float]]:
        names = list(map(str, feature_names)); n = len(names)
        matrix = [[0.0] * n for _ in range(n)]
        for i, a in enumerate(names):
            ea = self.entity_for_feature(a)
            for j, b in enumerate(names):
                if i == j:
                    matrix[i][j] = 1.0; continue
                eb = self.entity_for_feature(b)
                matrix[i][j] = self.relatedness(ea, eb) if ea and eb else 0.0
        return matrix

    def primary_group(self, feature: str) -> Optional[str]:
        groups = self.feature_groups.get(str(feature))
        if groups is None:
            mapped = self._normalized_index(self.feature_groups).get(
                normalize_ontology_name(feature)
            )
            groups = self.feature_groups.get(mapped, []) if mapped else []
        return groups[0] if groups else None


    def relationship_records(self) -> List[Dict[str, Any]]:
        """Relações OWL auditáveis, sem duplicar arestas simétricas."""
        rows: List[Dict[str, Any]] = []
        seen: Set[Tuple[str, str, str]] = set()
        for (left, right), rels in self.edge_types.items():
            for relation in sorted(rels):
                key = tuple(sorted((str(left), str(right)))) + (str(relation),)
                if key in seen:
                    continue
                seen.add(key)
                rows.append({
                    "type": "semantic_graph",
                    "source": str(left),
                    "target": str(right),
                    "relation": str(relation),
                    "weight": float(RELATION_WEIGHTS.get(str(relation), 0.50)),
                })
        return rows

    def summary(self) -> Dict[str, Any]:
        undirected_edges = sum(len(v) for v in self.edges.values()) // 2
        relation_counts = defaultdict(int)
        visited = set()
        for (a, b), rels in self.edge_types.items():
            key = tuple(sorted((a, b)))
            if key in visited: continue
            visited.add(key)
            for r in rels: relation_counts[r] += 1
        kind_counts = defaultdict(int)
        for kind in self.entity_kinds.values():
            kind_counts[str(kind)] += 1
        return {
            "nodes": len(self.nodes),
            "edges": int(undirected_edges),
            "mapped_features": len(self.feature_to_entity),
            "features_with_groups": sum(bool(v) for v in self.feature_groups.values()),
            "class_count": int(kind_counts.get("class", 0)),
            "datatype_property_count": int(kind_counts.get("datatype_property", 0)),
            "object_property_count": int(kind_counts.get("object_property", 0)),
            "annotation_property_count": int(kind_counts.get("annotation_property", 0)),
            "relation_counts": dict(relation_counts),
            "relation_weights": dict(RELATION_WEIGHTS),
            "is_active": bool(self.is_active),
            "load_error": self.load_error,
            "bounded_entities": len(self.entity_bounds),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": sorted(self.nodes),
            "edges": {k: sorted(v) for k, v in self.edges.items()},
            "labels": dict(self.labels),
            "feature_to_entity": dict(self.feature_to_entity),
            "feature_groups": dict(self.feature_groups),
            "entity_kinds": dict(self.entity_kinds),
            "summary": self.summary(),
        }


@dataclass
class OntologyDomainConstraints:
    """Validação e projecção de amostras para o domínio OWL (por índice)."""

    specs: Dict[int, Dict[str, Any]]
    n_features: int
    skipped_features: List[str] = field(default_factory=list)

    @property
    def is_active(self) -> bool:
        return bool(self.specs)

    def sample_validity_mask(self, X, feature_names: Optional[Sequence[str]] = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        valid = np.ones(len(X), dtype=bool)
        if X.ndim != 2:
            return valid
        for j, spec in self.specs.items():
            if j >= X.shape[1]:
                continue
            column = X[:, j]
            if spec["low"] is not None:
                valid &= ~(column < spec["low"])
            if spec["high"] is not None:
                valid &= ~(column > spec["high"])
            if spec["integer"]:
                valid &= np.isclose(column, np.round(column), atol=1e-9)
        return valid

    def project(self, X) -> np.ndarray:
        """Projecção mínima para o espaço conceptual válido.

        Arredonda features xsd:integer e recorta aos limites declarados; as
        restantes colunas ficam intactas.
        """
        out = np.array(X, dtype=float, copy=True)
        for j, spec in self.specs.items():
            if j >= out.shape[1]:
                continue
            column = out[:, j]
            if spec["integer"]:
                column = np.round(column)
            if spec["low"] is not None or spec["high"] is not None:
                low = -np.inf if spec["low"] is None else spec["low"]
                high = np.inf if spec["high"] is None else spec["high"]
                column = np.clip(column, low, high)
            out[:, j] = column
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "constrained_features": [spec["feature"] for spec in self.specs.values()],
            "integer_features": [s["feature"] for s in self.specs.values() if s["integer"]],
            "skipped_unit_mismatch": list(self.skipped_features),
        }


__all__ = [
    "OntologySemanticGraph", "OntologyDomainConstraints", "RELATION_WEIGHTS",
    "normalize_ontology_name",
]
