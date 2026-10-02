"""
Seleção de features ontológicas (top-k) antes de MLP_Onto e Trepan Reloaded.
Mantém sempre features ARFF originais.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.feature_selection import mutual_info_classif


ONTO_PREFIX = "onto_"


def _is_onto_feature(name: str, base_names: Sequence[str]) -> bool:
    if name in base_names:
        return False
    return name.startswith(ONTO_PREFIX) or name not in set(base_names)


def _drop_near_constant(X: np.ndarray, threshold: float = 1e-4) -> np.ndarray:
    if X.size == 0:
        return np.ones(X.shape[1], dtype=bool)
    std = np.std(X, axis=0)
    return std > threshold


def select_top_ontology_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    base_feature_names: List[str],
    top_k: Optional[int] = None,
    method: str = "mutual_info",
) -> Tuple[np.ndarray, List[str], dict]:
    """
    Seleciona top_k features ontológicas por MI; mantém todas as originais.
    top_k=None → mantém todas (após remover constantes).
    """
    keep_indices, names_out, info = fit_ontology_feature_selector(
        X, y, feature_names, base_feature_names, top_k=top_k, method=method
    )
    return apply_ontology_feature_selection(X, keep_indices), names_out, info


def fit_ontology_feature_selector(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    base_feature_names: List[str],
    top_k: Optional[int] = None,
    method: str = "mutual_info",
) -> Tuple[List[int], List[str], dict]:
    """Ajusta a selecao exclusivamente nos dados de treino.

    Devolve indices reutilizaveis em validacao/teste. Separar ``fit`` de
    ``apply`` impede que distribuicoes ou rotulos do teste influenciem a
    escolha de features ontologicas.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    names = list(feature_names)
    base_set = set(base_feature_names)

    base_indices = [i for i, n in enumerate(names) if n in base_set]
    onto_indices = [i for i, n in enumerate(names) if _is_onto_feature(n, base_feature_names)]

    if not onto_indices:
        return list(range(X.shape[1])), names, {
            "selected_onto": 0, "total_onto": 0, "method": method,
            "fit_scope": "training_only",
        }

    X_onto = X[:, onto_indices]
    active_mask = _drop_near_constant(X_onto)
    filtered_onto_indices = [onto_indices[i] for i, ok in enumerate(active_mask) if ok]

    n_removed = len(onto_indices) - len(filtered_onto_indices)
    if n_removed:
        print(
            f"[INFO] Removidas {n_removed} features ontológicas quase constantes."
        )

    if top_k is None or top_k <= 0 or len(filtered_onto_indices) <= top_k:
        keep_indices = sorted(set(base_indices + filtered_onto_indices))
        names_out = [names[i] for i in keep_indices]
        return keep_indices, names_out, {
            "selected_onto": len(filtered_onto_indices),
            "total_onto": len(onto_indices),
            "method": method,
            "top_k": top_k,
            "selected_indices": keep_indices,
            "fit_scope": "training_only",
        }

    X_onto_active = X[:, filtered_onto_indices]
    if method == "mutual_info":
        scores = mutual_info_classif(X_onto_active, y, random_state=42)
    else:
        scores = mutual_info_classif(X_onto_active, y, random_state=42)

    ranked = np.argsort(scores)[::-1][:top_k]
    top_onto_indices = [filtered_onto_indices[i] for i in ranked]

    keep_indices = sorted(set(base_indices + top_onto_indices))
    names_out = [names[i] for i in keep_indices]

    print(
        f"[INFO] Seleção onto: top_k={top_k}, "
        f"{len(top_onto_indices)}/{len(onto_indices)} features ontológicas mantidas."
    )

    return keep_indices, names_out, {
        "selected_onto": len(top_onto_indices),
        "total_onto": len(onto_indices),
        "method": method,
        "top_k": top_k,
        "top_scores": [float(scores[i]) for i in ranked[:5]],
        "selected_indices": keep_indices,
        "fit_scope": "training_only",
    }


def apply_ontology_feature_selection(X: np.ndarray, keep_indices: Sequence[int]) -> np.ndarray:
    """Aplica uma selecao previamente ajustada, sem voltar a consultar ``y``."""
    X = np.asarray(X, dtype=float)
    indices = [int(i) for i in keep_indices]
    if any(i < 0 or i >= X.shape[1] for i in indices):
        raise ValueError(
            "Indices de selecao ontologica incompatíveis com a matriz recebida."
        )
    return X[:, indices]


def audit_ontology_features(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    base_feature_names: Sequence[str],
    *,
    provenance: Optional[dict] = None,
    near_constant_threshold: float = 1e-4,
) -> dict:
    """Audita features OWL antes de qualquer seleção.

    Reporta constantes/quase constantes, duplicadas, valores ausentes, valores
    distintos e proveniência. A função não consulta teste externo.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    names = [str(n) for n in feature_names]
    base = set(str(n) for n in base_feature_names)
    provenance = dict(provenance or {})
    rows = []
    onto_indices = [i for i, n in enumerate(names) if n not in base]
    duplicate_of = {}
    for pos, i in enumerate(onto_indices):
        for j in onto_indices[:pos]:
            if np.array_equal(X[:, i], X[:, j], equal_nan=True):
                duplicate_of[i] = names[j]
                break
    for i in onto_indices:
        col = X[:, i]
        finite = col[np.isfinite(col)]
        std = float(np.std(finite)) if len(finite) else 0.0
        distinct = int(len(np.unique(finite))) if len(finite) else 0
        meta = provenance.get(names[i]) or {}
        if isinstance(meta, str):
            meta = {"owl_concept": meta}
        missing_pct = float(np.mean(~np.isfinite(col))) if len(col) else 0.0
        constant_pct = 1.0
        if len(finite):
            _, counts = np.unique(finite, return_counts=True)
            constant_pct = float(counts.max() / len(finite))
        rows.append({
            "name": names[i],
            "owl_concept": meta.get("owl_concept"),
            "owl_property": meta.get("owl_property"),
            "derivation_rule": meta.get("derivation_rule"),
            "feature_type": meta.get("feature_type", "numeric"),
            "missing_pct": missing_pct,
            "constant_pct": constant_pct,
            "n_distinct": distinct,
            "near_constant": bool(std <= near_constant_threshold),
            "duplicate_of": duplicate_of.get(i),
            "has_provenance": bool(meta),
            "fit_scope": "development_only",
        })
    return {"features": rows, "n_ontology_features": len(rows), "test_used": False}


def filter_audited_ontology_features(audit: dict) -> List[str]:
    """Rejeita features constantes/quase constantes, duplicadas ou sem proveniência."""
    accepted = []
    for row in audit.get("features", []):
        if row.get("near_constant"):
            continue
        if row.get("duplicate_of"):
            continue
        if not row.get("has_provenance"):
            continue
        accepted.append(str(row["name"]))
    return accepted
