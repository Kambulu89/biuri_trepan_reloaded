"""Contratos explícitos dos espaços de features do BIURI/TREPAN Reloaded."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence
import hashlib
import json
import numpy as np


def _sha256_names(names: Sequence[str]) -> str:
    payload = json.dumps([str(x) for x in names], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class FeatureOrigin:
    name: str
    origin: str
    owl_concept: Optional[str] = None
    owl_property: Optional[str] = None
    derivation_rule: Optional[str] = None


@dataclass(frozen=True)
class FeatureSpaceManifest:
    space_name: str
    feature_names: List[str]
    n_features: int
    schema_hash: str
    transformer_id: str
    feature_origins: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_feature_space_manifest(
    space_name: str,
    feature_names: Sequence[str],
    *,
    transformer_id: str,
    origins: Optional[Mapping[str, Mapping[str, Any] | str]] = None,
) -> FeatureSpaceManifest:
    names = [str(x) for x in feature_names]
    if len(names) != len(set(names)):
        duplicates = sorted({x for x in names if names.count(x) > 1})
        raise ValueError(f"Schema inválido: features duplicadas: {duplicates}")
    if space_name not in {"original_space", "ontological_space", "residual_space"}:
        raise ValueError(f"Espaço de features desconhecido: {space_name}")
    out = []
    origins = dict(origins or {})
    for name in names:
        meta = origins.get(name, "original" if space_name == "original_space" else "unknown")
        if isinstance(meta, str):
            meta = {"origin": meta}
        row = {"name": name, **dict(meta)}
        row.setdefault("origin", "unknown")
        out.append(row)
    return FeatureSpaceManifest(
        space_name=space_name,
        feature_names=names,
        n_features=len(names),
        schema_hash=_sha256_names(names),
        transformer_id=str(transformer_id),
        feature_origins=out,
    )


def validate_feature_matrix(X, manifest: FeatureSpaceManifest, *, feature_names: Optional[Sequence[str]] = None) -> None:
    matrix = np.asarray(X)
    if matrix.ndim != 2:
        raise ValueError("A matriz de features deve ser bidimensional.")
    if matrix.shape[1] != manifest.n_features:
        raise ValueError(
            f"Schema incompatível para {manifest.space_name}: esperado {manifest.n_features} features, recebido {matrix.shape[1]}."
        )
    if feature_names is not None:
        names = [str(x) for x in feature_names]
        if names != manifest.feature_names:
            raise ValueError(
                f"Ordem/nome das features incompatível com {manifest.space_name}; alinhamento posicional silencioso é proibido."
            )


def assert_original_space_has_no_ontology(manifest: FeatureSpaceManifest) -> None:
    if manifest.space_name != "original_space":
        raise ValueError("Manifesto fornecido não representa original_space.")
    bad = [r["name"] for r in manifest.feature_origins if str(r.get("origin", "")).lower() not in {"original", "raw", "dataset"}]
    if bad:
        raise ValueError(f"MLP Original não pode receber features ontológicas/residuais: {bad}")


def assert_ontological_space_enriches_original(original: FeatureSpaceManifest, ontological: FeatureSpaceManifest) -> None:
    if original.space_name != "original_space" or ontological.space_name != "ontological_space":
        raise ValueError("Espaços inválidos para validação original→ontológico.")
    if ontological.feature_names[: len(original.feature_names)] != original.feature_names:
        raise ValueError("O espaço ontológico deve preservar as features originais, na mesma ordem, antes do enriquecimento.")
    if len(ontological.feature_names) <= len(original.feature_names):
        raise ValueError("O espaço ontológico aceite deve conter ao menos uma feature semântica além das originais.")


__all__ = [
    "FeatureOrigin", "FeatureSpaceManifest", "build_feature_space_manifest",
    "validate_feature_matrix", "assert_original_space_has_no_ontology",
    "assert_ontological_space_enriches_original",
]
