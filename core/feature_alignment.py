"""Alinhamento estrito e nominal de espaços de atributos.

Nunca corta colunas nem inventa zeros. Uma incompatibilidade de schema é um
erro de dados/modelo e deve interromper o pipeline antes de produzir métricas.
"""
from __future__ import annotations

from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np

OracleLike = Any


def oracle_n_features(oracle_model: Optional[OracleLike]) -> Optional[int]:
    if oracle_model is None:
        return None
    n = getattr(oracle_model, "n_features_in_", None)
    if n is not None:
        return int(n)
    steps = getattr(oracle_model, "named_steps", None)
    if steps:
        for step in steps.values():
            n = getattr(step, "n_features_in_", None)
            if n is not None:
                return int(n)
    return None


def infer_oracle_type(oracle_model: Optional[OracleLike], explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    if getattr(oracle_model, "ORACLE_TYPE", None) == "residual_ontological":
        return "MLP Residual Ontológico"
    meta = getattr(oracle_model, "arff_meta", None)
    if isinstance(meta, dict):
        space = meta.get("oracle_space")
        if space == "augmented":
            return "MLP_Onto"
        if space == "residual_ontological":
            return "MLP Residual Ontológico"
        if space == "original":
            return "MLP Original"
    n = oracle_n_features(oracle_model)
    if n is not None and n > 12:
        return "MLP_Onto"
    return "MLP Original"


def _normalize_matrix_names(
    feature_names: Optional[Sequence[str]], n_columns: int
) -> List[str]:
    if not feature_names:
        return []
    names = [str(name) for name in feature_names]
    if len(names) != n_columns:
        raise ValueError(
            f"Schema inválido: foram declarados {len(names)} nomes para "
            f"uma matriz com {n_columns} colunas."
        )
    if len(set(names)) != len(names):
        raise ValueError("Schema inválido: nomes de atributos duplicados.")
    return names


def _is_onto_feature(name: str) -> bool:
    return str(name).startswith("onto_")


def _log_feature_alignment(
    oracle_type: str,
    n_oracle: int,
    n_target: int,
    strategy: str,
    result_shape: Tuple[int, ...],
) -> None:
    print("\n[FEATURE ALIGNMENT]")
    print(f"Oracle:\n  type: {oracle_type}\n  n_features: {n_oracle}")
    print(f"Target matrix:\n  n_features: {n_target}")
    print(f"Strategy:\n  {strategy}")
    print(f"Result:\n  aligned matrix shape: {result_shape}")


def _indices_by_names(
    matrix_names: Sequence[str],
    projection_names: Sequence[str],
    n_oracle: int,
) -> Optional[List[int]]:
    name_to_idx = {n: i for i, n in enumerate(matrix_names)}
    indices = [name_to_idx[n] for n in projection_names if n in name_to_idx]
    if len(indices) == n_oracle:
        return indices
    if len(indices) > n_oracle:
        return indices[:n_oracle]
    return None


def align_feature_spaces(
    oracle_model: Optional[OracleLike],
    X_target,
    feature_names: Optional[Sequence[str]] = None,
    original_feature_names: Optional[Sequence[str]] = None,
    oracle_type: Optional[str] = None,
    n_oracle_features: Optional[int] = None,
    log: bool = True,
) -> np.ndarray:
    """
    Alinha X_target ao espaço de features esperado pelo oráculo MLP.

    - Dimensões e ordem iguais → devolve X_target
    - Matriz enriquecida → projeção apenas por nomes originais explícitos
    - Incompatibilidade/ambiguidade → ``ValueError`` (fail-fast)
    """
    if X_target is None:
        return X_target

    X = np.asarray(X_target, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X_target deve ser 2D; recebido shape={X.shape}.")
    n_target = int(X.shape[1])
    n_oracle = n_oracle_features if n_oracle_features is not None else oracle_n_features(oracle_model)
    label = infer_oracle_type(oracle_model, oracle_type)

    if n_oracle is None:
        raise ValueError(
            "Não foi possível determinar n_features_in_ do oráculo. "
            "Associe um ModelBundle/schema explícito antes da previsão."
        )

    if n_oracle == n_target:
        if log:
            _log_feature_alignment(label, n_oracle, n_target, "Direct", X.shape)
        return X

    matrix_names = _normalize_matrix_names(feature_names, n_target)
    if not matrix_names:
        raise ValueError(
            f"O oráculo espera {n_oracle} features e a matriz contém {n_target}; "
            "a projeção exige feature_names explícitos."
        )
    projection_candidates: List[Sequence[str]] = []

    if original_feature_names:
        projection_candidates.append(list(original_feature_names))
    if feature_names and len(feature_names) == n_oracle:
        projection_candidates.append(list(feature_names))

    non_onto = [n for n in matrix_names if not _is_onto_feature(n)]
    if len(non_onto) == n_oracle:
        projection_candidates.append(non_onto)

    seen = set()
    for cand in projection_candidates:
        key = tuple(cand)
        if key in seen:
            continue
        seen.add(key)
        indices = _indices_by_names(matrix_names, cand, n_oracle)
        if indices is not None:
            aligned = X[:, indices]
            if log:
                _log_feature_alignment(
                    label,
                    n_oracle,
                    n_target,
                    "Projection using feature names",
                    aligned.shape,
                )
            return aligned

    raise ValueError(
        f"Espaços de atributos incompatíveis: oráculo={label} espera {n_oracle}, "
        f"matriz tem {n_target}. Não foi encontrada projeção nominal completa."
    )
