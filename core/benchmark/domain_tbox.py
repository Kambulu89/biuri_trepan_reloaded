"""Geração genérica de TBoxes OWL de domínio (sem instâncias) a partir de grupos de features.

Agnóstico ao dataset: recebe um nome qualquer, grupos de features e relações opcionais. Os catálogos específicos de
datasets de validação vivem em ``validation/`` e não fazem parte do núcleo científico.
"""
from __future__ import annotations

import re
import types
from pathlib import Path
from typing import Dict, List, Optional


def _safe_name(label: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", label)
    return "has" + "".join(word[:1].upper() + word[1:] for word in words)


def _inject_feature_relation_annotations(
    destination: Path,
    feature_relations: Optional[Dict[str, Dict[str, str]]],
) -> None:
    """Anexa metadados de relação ao RDF/XML sem depender da API de annotations.

    O Owlready2 continua responsável por serializar uma TBox válida; esta etapa
    apenas acrescenta ``owl:AnnotationProperty`` e dois literais por propriedade
    quando explicitamente fornecidos pelo catálogo de domínio.
    """
    if not feature_relations:
        return
    text = destination.read_text(encoding="utf-8")
    if 'rdf:about="#measurementFamily"' not in text:
        text = text.replace(
            "</owl:Ontology>",
            "</owl:Ontology>\n\n"
            '<owl:AnnotationProperty rdf:about="#measurementFamily"/>\n'
            '<owl:AnnotationProperty rdf:about="#statisticRole"/>',
            1,
        )

    def enrich(block_match):
        block = block_match.group(0)
        label_match = re.search(r"<rdfs:label[^>]*>(.*?)</rdfs:label>", block, re.S)
        if not label_match or "<measurementFamily" in block:
            return block
        label = re.sub(r"<.*?>", "", label_match.group(1)).strip()
        relation = feature_relations.get(label) or {}
        family = relation.get("measurement_family")
        role = relation.get("statistic_role")
        if not family or not role:
            return block
        annotation = (
            '  <measurementFamily rdf:datatype="http://www.w3.org/2001/XMLSchema#string">'
            f"{family}</measurementFamily>\n"
            '  <statisticRole rdf:datatype="http://www.w3.org/2001/XMLSchema#string">'
            f"{role}</statisticRole>\n"
        )
        return block.replace("</owl:DatatypeProperty>", annotation + "</owl:DatatypeProperty>")

    text = re.sub(
        r"<owl:DatatypeProperty\b.*?</owl:DatatypeProperty>",
        enrich,
        text,
        flags=re.S,
    )
    destination.write_text(text, encoding="utf-8")


def create_domain_tbox(
    path,
    *,
    dataset_name: str,
    groups: Dict[str, List[str]],
    feature_relations: Optional[Dict[str, Dict[str, str]]] = None,
) -> str:
    """Gera RDF/XML TBox com domínios sem criar qualquer NamedIndividual."""
    from owlready2 import AllDisjoint, DataProperty, Thing, World

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    iri = f"http://example.org/trepan-benchmark/{dataset_name.replace('_', '-')}#"
    world = World()
    ontology = world.get_ontology(iri)
    with ontology:
        measurement = types.new_class("DomainMeasurement", (Thing,))
        measurement.label = [f"{dataset_name} domain measurement"]
        group_classes = []
        used_names = set()
        for group_name, feature_names in groups.items():
            group = types.new_class(group_name, (measurement,))
            group.label = [group_name]
            group_classes.append(group)
            for feature_name in feature_names:
                property_name = _safe_name(feature_name)
                if property_name in used_names:
                    raise ValueError(f"Nome OWL duplicado: {property_name}")
                used_names.add(property_name)
                prop = types.new_class(property_name, (DataProperty,))
                prop.label = [str(feature_name)]
                prop.domain = [group]
                prop.range = [float]
        if len(group_classes) > 1:
            AllDisjoint(group_classes)
    ontology.metadata.comment.append(
        "TBox de benchmark: contém semântica de domínio e zero instâncias do dataset."
    )
    ontology.metadata.comment.append(
        "A estrutura foi definida apenas a partir da descrição das features; "
        "rótulos, partições e resultados do benchmark não foram consultados."
    )
    ontology.save(file=str(destination), format="rdfxml")
    _inject_feature_relation_annotations(destination, feature_relations)
    return str(destination)


__all__ = ["create_domain_tbox"]
