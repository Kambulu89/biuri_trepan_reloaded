"""TBoxes OWL 2 de domínio para o benchmark de ablação.

Não contêm linhas, pacientes ou amostras: apenas classes, propriedades e
rótulos correspondentes às definições públicas dos datasets scikit-learn.
"""
from __future__ import annotations

import types
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


IRIS_GROUPS = {
    "SepalMeasurement": ["sepal length (cm)", "sepal width (cm)"],
    "PetalMeasurement": ["petal length (cm)", "petal width (cm)"],
}

WINE_GROUPS = {
    "AcidityProfile": ["malic_acid", "ash", "alcalinity_of_ash"],
    "PhenolicProfile": [
        "total_phenols", "flavanoids", "nonflavanoid_phenols",
        "proanthocyanins", "od280/od315_of_diluted_wines",
    ],
    "PhysicalChemicalProfile": [
        "alcohol", "magnesium", "color_intensity", "hue", "proline",
    ],
}

DIABETES_GROUPS = {
    "DemographicMeasurement": ["age", "sex"],
    "AnthropometricMeasurement": ["bmi", "bp"],
    "SerumMeasurement": ["s1", "s2", "s3", "s4", "s5", "s6"],
}


def digits_groups(feature_names: Iterable[str]) -> Dict[str, List[str]]:
    """Agrupa pixels pela região espacial, sem consultar qualquer rótulo."""
    groups = {"UpperImageRegion": [], "MiddleImageRegion": [], "LowerImageRegion": []}
    for feature in feature_names:
        parts = str(feature).rsplit("_", 2)
        try:
            row = int(parts[-2])
        except (ValueError, IndexError):
            row = 3
        key = "UpperImageRegion" if row <= 2 else "MiddleImageRegion" if row <= 5 else "LowerImageRegion"
        groups[key].append(str(feature))
    return groups


def breast_cancer_groups(feature_names: Iterable[str]) -> Dict[str, List[str]]:
    groups = {"MeanMorphology": [], "ErrorMorphology": [], "WorstMorphology": []}
    for feature in feature_names:
        if feature.startswith("mean "):
            groups["MeanMorphology"].append(feature)
        elif feature.endswith(" error"):
            groups["ErrorMorphology"].append(feature)
        elif feature.startswith("worst "):
            groups["WorstMorphology"].append(feature)
    return groups


def breast_cancer_family_roles(feature_names: Iterable[str]) -> Dict[str, Dict[str, str]]:
    """Metadados TBox transversais: mesma medição, papéis mean/error/worst.

    A informação vem apenas do significado público dos nomes das features; não
    consulta rótulos, partições ou resultados preditivos. O motor de features
    continua agnóstico: ele apenas lê ``measurementFamily``/``statisticRole``.
    """
    relations: Dict[str, Dict[str, str]] = {}
    for raw in feature_names:
        feature = str(raw)
        if feature.startswith("mean "):
            family, role = feature[len("mean "):], "mean"
        elif feature.endswith(" error"):
            family, role = feature[:-len(" error")], "error"
        elif feature.startswith("worst "):
            family, role = feature[len("worst "):], "worst"
        else:
            continue
        relations[feature] = {
            "measurement_family": family.strip().replace(" ", "_"),
            "statistic_role": role,
        }
    return relations


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


def ensure_builtin_domain_ontologies(output_dir) -> Dict[str, str]:
    from sklearn.datasets import load_breast_cancer, load_digits

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    breast_features = list(map(str, load_breast_cancer().feature_names))
    catalog = {
        "iris": (IRIS_GROUPS, None),
        "wine": (WINE_GROUPS, None),
        "breast_cancer": (
            breast_cancer_groups(breast_features),
            breast_cancer_family_roles(breast_features),
        ),
        "diabetes_progression": (DIABETES_GROUPS, None),
        "digits": (digits_groups(load_digits().feature_names), None),
    }
    return {
        name: create_domain_tbox(
            root / f"{name}.owl", dataset_name=name, groups=groups,
            feature_relations=relations,
        )
        for name, (groups, relations) in catalog.items()
    }
