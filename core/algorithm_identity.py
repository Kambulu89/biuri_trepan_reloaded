"""Identidade científica dos algoritmos e dos espaços de atributos."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict


class FeatureSpace(str, Enum):
    ORIGINAL = "original"
    ENRICHED = "enriched"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class AlgorithmIdentity:
    key: str
    display_name: str
    implementation: str
    oracle: str
    feature_space: FeatureSpace
    canonical: bool


ALGORITHMS: Dict[str, AlgorithmIdentity] = {
    "mlp_original": AlgorithmIdentity(
        "mlp_original", "MLP Original", "scikit-learn MLPClassifier",
        "ground-truth labels", FeatureSpace.ORIGINAL, True,
    ),
    "c45_j48": AlgorithmIdentity(
        "c45_j48", "C4.5-Nativo",
        "Gain Ratio + valores ausentes fracionários + poda pessimista",
        "ground-truth labels", FeatureSpace.ORIGINAL, True,
    ),
    "trepan_original": AlgorithmIdentity(
        "trepan_original", "TREPAN Original",
        "best-first + testes m-of-n + consultas ao MLP Original (core.trepan_original)", "MLP Original",
        FeatureSpace.ORIGINAL, True,
    ),
    "trepan_reloaded": AlgorithmIdentity(
        "trepan_reloaded", "TREPAN Reloaded",
        "núcleo TREPAN histórico + enriquecimento/semântica OWL opcional",
        "oráculo ativo identificado por execução", FeatureSpace.ENRICHED, True,
    ),
}


def identity_for(key: str) -> AlgorithmIdentity:
    try:
        return ALGORITHMS[key]
    except KeyError as exc:
        raise ValueError(f"Algoritmo desconhecido: {key!r}") from exc
