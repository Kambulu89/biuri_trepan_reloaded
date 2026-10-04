"""Catálogo de TBoxes de domínio dos datasets de VALIDAÇÃO (fora do núcleo científico).

Não contêm linhas, pacientes ou amostras: apenas classes, propriedades e
rótulos correspondentes às definições públicas dos datasets scikit-learn.
"""
from __future__ import annotations

import types
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from core.benchmark.domain_tbox import create_domain_tbox


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
