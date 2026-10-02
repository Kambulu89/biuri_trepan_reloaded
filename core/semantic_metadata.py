"""Metadata semântica por feature e matriz de relatedness para o TREPAN Reloaded.

Cada feature (original ou derivada ``onto_*``) recebe um registo rastreável:
entidade OWL, classe, família, papel estatístico, fontes, profundidade, relações.

A matriz de relatedness usa **regras ontológicas**, nunca nomes de datasets:

====================================  =========================
relação entre as duas features        relatedness (configurável)
====================================  =========================
mesma entidade OWL                    ``same_entity``       = 1.00
derivada ↔ uma das suas fontes        ``derived_to_source`` = 0.90
mesma família de medida               ``same_family``       = 0.75
mesma superclasse / grupo             ``same_superclass``   = 0.50
relação distante no grafo             ``graph_scale`` x relatedness do grafo (< same_superclass)
sem relação                           0.00
====================================  =========================

Dupla contagem (Parte 22): uma feature ``onto_*`` já *contém* a relação semântica que a
gerou. Por isso o peso de *bias* de split de uma feature derivada nunca excede o maior peso
das suas fontes; o conhecimento entra uma vez como feature e, no split, apenas através da
coesão entre literais (relatedness), não também por um peso extra.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RelatednessConfig:
    same_entity: float = 1.0
    derived_to_source: float = 0.90
    same_family: float = 0.75
    same_superclass: float = 0.50
    graph_scale: float = 0.40
    bias_weight_per_confidence: float = 0.25  # peso de split = 1 + k * score do matching

    def validate(self) -> None:
        values = (self.same_entity, self.derived_to_source, self.same_family,
                  self.same_superclass, self.graph_scale)
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError("RelatednessConfig: todos os valores têm de estar em [0, 1].")
        if not (self.same_entity >= self.derived_to_source >= self.same_family >= self.same_superclass):
            raise ValueError("RelatednessConfig: a ordem same_entity >= derived_to_source >= "
                             "same_family >= same_superclass tem de ser respeitada.")


def build_semantic_metadata(
    feature_names: Sequence[str],
    processor=None,
    graph=None,
    matches: Optional[Sequence[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Um registo por feature, na mesma ordem de ``feature_names``."""
    names = [str(n) for n in feature_names]
    entity_of = dict(getattr(processor, "column_entity_map_", {}) or {})
    score_of = {str(m.get("feature")): float(m.get("score", 0.0)) for m in (matches or []) if m.get("accepted", True)}
    audit = {row["feature"]: row for row in (getattr(processor, "feature_audit_", []) or [])
             if row.get("accepted")}
    specs = {s["name"]: s for s in (getattr(processor, "feature_specs_", []) or [])}
    relations = graph.relationship_records() if graph is not None and hasattr(graph, "relationship_records") else []

    def graph_depth(feature: str) -> float:
        try:
            return float(graph.get_feature_depth(feature)) if graph is not None else 0.0
        except Exception:
            return 0.0

    def graph_group(feature: str) -> Optional[str]:
        try:
            return graph.primary_group(feature) if graph is not None else None
        except Exception:
            return None

    out: List[Dict[str, Any]] = []
    for name in names:
        if name in specs:
            spec, row = specs[name], audit.get(name, {})
            sources = [s for s in (spec.get("sources") or ([spec["source"]] if spec.get("source") else []))]
            entities = list(row.get("owl_origin", {}).get("owl_entities_or_properties", []))
            primary_entity = entities[0] if entities else None
            source_groups = [graph_group(s) for s in sources if graph_group(s)]
            out.append({
                "feature_name": name, "origin": "ontology", "kind": spec.get("kind"),
                "ontology_entity": primary_entity, "ontology_entities": entities,
                "ontology_class": spec.get("family") or (source_groups[0] if source_groups else None),
                "ontology_family": spec.get("family"),
                "statistic_role": "/".join(spec.get("roles") or []) or None,
                "source_features": sources,
                "reasoner_inferred": bool(spec.get("reasoner_inferred", False)),
                "knowledge_source": row.get("knowledge_source"),
                "semantic_depth": max([graph_depth(s) for s in sources], default=0.0) + 1.0,
                "semantic_relations": [],
                "matching_score": None,
            })
        else:
            entity = entity_of.get(name)
            out.append({
                "feature_name": name, "origin": "original", "kind": "original",
                "ontology_entity": entity, "ontology_entities": [entity] if entity else [],
                "ontology_class": graph_group(name),
                "ontology_family": None, "statistic_role": None, "source_features": [],
                "reasoner_inferred": False, "knowledge_source": "ontology" if entity else None,
                "semantic_depth": graph_depth(name),
                "semantic_relations": [r for r in relations if entity and entity in (r["source"], r["target"])][:10],
                "matching_score": score_of.get(name),
            })
    # famílias declaradas nas features originais (anotação na entidade) herdam-se das derivadas
    families = {}
    for meta in out:
        if meta["origin"] == "ontology" and meta["ontology_family"]:
            for src in meta["source_features"]:
                families.setdefault(src, meta["ontology_family"])
    for meta in out:
        if meta["origin"] == "original" and meta["feature_name"] in families:
            meta["ontology_family"] = families[meta["feature_name"]]
    return out


def build_relatedness_matrix(
    metadata: Sequence[Dict[str, Any]],
    graph=None,
    config: RelatednessConfig = RelatednessConfig(),
) -> np.ndarray:
    """Matriz simétrica [0,1] com diagonal 1, pelas regras da tabela do módulo."""
    config.validate()
    n = len(metadata)
    matrix = np.zeros((n, n), dtype=float)
    for i in range(n):
        matrix[i, i] = 1.0
    for i in range(n):
        a = metadata[i]
        for j in range(i + 1, n):
            b = metadata[j]
            value = 0.0
            ent_a, ent_b = set(a["ontology_entities"]), set(b["ontology_entities"])
            if ent_a and ent_b and ent_a == ent_b:
                value = config.same_entity
            if a["feature_name"] in b["source_features"] or b["feature_name"] in a["source_features"]:
                value = max(value, config.derived_to_source)
            if a["ontology_family"] and a["ontology_family"] == b["ontology_family"]:
                value = max(value, config.same_family)
            if a["ontology_class"] and a["ontology_class"] == b["ontology_class"]:
                value = max(value, config.same_superclass)
            if value == 0.0 and graph is not None and a["ontology_entities"] and b["ontology_entities"]:
                try:
                    rel = float(graph.relatedness(a["ontology_entities"][0], b["ontology_entities"][0]))
                except Exception:
                    rel = 0.0
                value = min(config.graph_scale * rel, config.same_superclass * 0.99)
            matrix[i, j] = matrix[j, i] = float(np.clip(value, 0.0, 1.0))
    return matrix


def semantic_split_weights(
    metadata: Sequence[Dict[str, Any]], config: RelatednessConfig = RelatednessConfig()
) -> np.ndarray:
    """Pesos de *bias* de split sem dupla contagem.

    Originais: ``1 + k * score`` do matching. Derivadas: no máximo o maior peso das fontes
    (a relação já está dentro da feature; não recebe bónus adicional por ser ``onto_*``).
    """
    by_name = {m["feature_name"]: i for i, m in enumerate(metadata)}
    weights = np.ones(len(metadata), dtype=float)
    for i, meta in enumerate(metadata):
        if meta["origin"] == "original":
            score = meta.get("matching_score")
            weights[i] = 1.0 + config.bias_weight_per_confidence * float(score) if score else 1.0
    for i, meta in enumerate(metadata):
        if meta["origin"] == "ontology":
            parents = [weights[by_name[s]] for s in meta["source_features"] if s in by_name]
            weights[i] = max(parents) if parents else 1.0
    return weights


def audit_double_counting(
    metadata: Sequence[Dict[str, Any]], weights: Sequence[float], tolerance: float = 1e-9
) -> List[Dict[str, Any]]:
    """Lista features derivadas cujo peso excede o das fontes (dupla contagem)."""
    by_name = {m["feature_name"]: i for i, m in enumerate(metadata)}
    flagged = []
    for i, meta in enumerate(metadata):
        if meta["origin"] != "ontology":
            continue
        parents = [weights[by_name[s]] for s in meta["source_features"] if s in by_name]
        ceiling = max(parents) if parents else 1.0
        if weights[i] > ceiling + tolerance:
            flagged.append({
                "feature": meta["feature_name"], "weight": float(weights[i]),
                "max_source_weight": float(ceiling),
                "reason": "derived_feature_already_encodes_relation",
            })
    return flagged


def apply_family_relatedness(
    matrix,
    source_columns: Sequence[Optional[str]],
    column_family: Dict[str, str],
    config: RelatednessConfig = RelatednessConfig(),
) -> np.ndarray:
    """Sobrepõe ``same_family`` à matriz existente (só aumenta, nunca reduz).

    ``source_columns[i]`` é a coluna original da feature do modelo ``i`` (várias features
    codificadas podem partilhar a mesma coluna); ``column_family`` vem das famílias que
    a OWL declara. Sem famílias declaradas a matriz devolvida é igual à recebida.
    """
    out = np.array(matrix, dtype=float, copy=True)
    n = len(source_columns)
    for i in range(n):
        fam_i = column_family.get(str(source_columns[i])) if source_columns[i] is not None else None
        if not fam_i:
            continue
        for j in range(i + 1, n):
            fam_j = column_family.get(str(source_columns[j])) if source_columns[j] is not None else None
            if fam_j == fam_i and source_columns[i] != source_columns[j]:
                out[i, j] = out[j, i] = max(out[i, j], config.same_family)
    return out


def extend_semantic_inputs(
    names: Sequence[str],
    weights,
    groups: Sequence[Optional[str]],
    entities: Sequence[Optional[str]],
    matrix,
    processor,
    selected: Sequence[str],
    config: RelatednessConfig = RelatednessConfig(),
):
    """Estende pesos/grupos/entidades/relatedness do espaço original ao aumentado.

    Cada feature ``onto_*`` selecionada entra no fim, por esta ordem. Regras:

    * peso de split = o maior peso das suas fontes (sem bónus extra por ser ``onto_*``);
    * grupo = a família OWL da feature, ou o grupo da primeira fonte;
    * relatedness com a fonte = ``derived_to_source``; com as restantes features herda o
      máximo das relações das suas fontes (limitado a ``derived_to_source``); entre duas
      derivadas que partilham fonte ou família = ``same_family``.

    Devolve ``(names, weights, groups, entities, matrix, records)``.
    """
    config.validate()
    names = [str(n) for n in names]
    n = len(names)
    weights = np.asarray(weights, dtype=float)
    base = np.asarray(matrix, dtype=float)
    specs = {s["name"]: s for s in (getattr(processor, "feature_specs_", []) or [])}
    entity_map = dict(getattr(processor, "column_entity_map_", {}) or {})
    selected = [f for f in selected if f in specs]
    m = n + len(selected)
    out = np.zeros((m, m), dtype=float)
    out[:n, :n] = base
    new_weights = list(weights)
    new_groups, new_entities, records, src_of = list(groups), list(entities), [], []
    for k, name in enumerate(selected):
        spec = specs[name]
        sources = [str(s) for s in (spec.get("sources") or ([spec["source"]] if spec.get("source") else []))]
        src_idx = [names.index(s) for s in sources if s in names]
        src_of.append(set(src_idx))
        new_weights.append(float(max((weights[i] for i in src_idx), default=1.0)))
        new_groups.append(spec.get("family") or next((groups[i] for i in src_idx if groups[i]), None))
        entity = next(iter(spec.get("owl_entities") or []), None) or entity_map.get(sources[0] if sources else "")
        new_entities.append(entity)
        row = n + k
        out[row, row] = 1.0
        for j in range(n):
            inherited = max((base[i, j] for i in src_idx if i != j), default=0.0)
            value = min(float(inherited), config.derived_to_source)
            if j in src_idx:
                value = config.derived_to_source
            out[row, j] = out[j, row] = value
        records.append({"feature": name, "sources": sources, "weight": new_weights[-1],
                        "group": new_groups[-1], "entity": entity})
    for a in range(len(selected)):
        for b in range(a + 1, len(selected)):
            same_family = specs[selected[a]].get("family") and specs[selected[a]].get("family") == specs[selected[b]].get("family")
            if src_of[a] & src_of[b] or same_family:
                out[n + a, n + b] = out[n + b, n + a] = config.same_family
    return names + selected, np.asarray(new_weights, dtype=float), new_groups, new_entities, out, records
