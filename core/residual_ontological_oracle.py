"""
MLP Residual Ontológico — oráculo composto: probabilidades base + correção ontológica.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold, StratifiedKFold


class ResidualOntologicalOracle:
    """Oráculo que combina MLP Original (base) com MLP Residual (onto_features)."""

    ORACLE_TYPE = "residual_ontological"
    ORACLE_LABEL = "MLP Residual Ontológico"

    def __init__(
        self,
        mlp_original,
        mlp_residual,
        original_column_indices: Sequence[int],
        onto_column_indices: Sequence[int],
        n_enriched_features: int,
    ):
        self.mlp_original = mlp_original
        self.mlp_residual = mlp_residual
        self.original_column_indices = list(original_column_indices)
        self.onto_column_indices = list(onto_column_indices)
        self.n_features_in_ = int(n_enriched_features)
        self.classes_ = getattr(
            mlp_residual, "classes_", getattr(mlp_original, "classes_", None)
        )

    def _build_residual_input(self, X_enriched) -> np.ndarray:
        X = np.asarray(X_enriched, dtype=float)
        X_orig = X[:, self.original_column_indices]
        onto = X[:, self.onto_column_indices]
        base_probs = self.mlp_original.predict_proba(X_orig)
        return np.hstack([base_probs, onto])

    def predict_proba(self, X_enriched):
        return self.mlp_residual.predict_proba(self._build_residual_input(X_enriched))

    def predict(self, X_enriched):
        return self.mlp_residual.predict(self._build_residual_input(X_enriched))


def original_column_indices(
    feature_names: Sequence[str], original_feature_names: Sequence[str]
) -> List[int]:
    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    missing = [name for name in original_feature_names if name not in name_to_idx]
    if missing:
        raise ValueError(
            "A matriz enriquecida não contém todos os atributos originais: "
            + ", ".join(map(str, missing[:8]))
        )
    return [name_to_idx[n] for n in original_feature_names]


def onto_column_indices(
    feature_names: Sequence[str], original_feature_names: Sequence[str]
) -> List[int]:
    orig_set = set(original_feature_names)
    return [
        i
        for i, name in enumerate(feature_names)
        if str(name).startswith("onto_") or name not in orig_set
    ]


def build_residual_feature_matrix(
    X_enriched,
    mlp_original,
    original_column_indices: Sequence[int],
    onto_column_indices: Sequence[int],
) -> np.ndarray:
    """Constrói X_residual = [base_probs, onto_features]."""
    X = np.asarray(X_enriched, dtype=float)
    X_orig = X[:, original_column_indices]
    onto = X[:, onto_column_indices]
    base_probs = mlp_original.predict_proba(X_orig)
    return np.hstack([base_probs, onto])


def build_oof_residual_feature_matrices(
    X_train_enriched,
    y_train,
    X_test_enriched,
    mlp_original,
    original_column_indices: Sequence[int],
    onto_column_indices: Sequence[int],
    n_splits: int = 5,
    random_state: int = 42,
    groups: Optional[Sequence] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Cria treino residual sem probabilidades *in-sample* do MLP base.

    O treino usa ``cross_val_predict``; validacao/teste usa o MLP original
    treinado apenas no conjunto de treino externo. Assim a segunda camada nao
    aprende a partir de previsoes excessivamente confiantes do proprio treino.
    """
    X_train = np.asarray(X_train_enriched, dtype=float)
    X_test = np.asarray(X_test_enriched, dtype=float)
    y_train = np.asarray(y_train)
    X_train_orig = X_train[:, original_column_indices]
    X_test_orig = X_test[:, original_column_indices]

    _, counts = np.unique(y_train, return_counts=True)
    if len(counts) < 2 or int(counts.min()) < 2:
        raise ValueError(
            "Sao necessarias pelo menos duas classes e duas amostras por classe "
            "para construir probabilidades out-of-fold."
        )
    folds = max(2, min(int(n_splits), int(counts.min())))
    if groups is not None:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != len(y_train):
            raise ValueError("groups deve ter uma entrada por amostra de treino.")
        unique_groups = np.unique(groups_arr)
        if len(unique_groups) < 2:
            raise ValueError("OOF por paciente/grupo exige pelo menos dois grupos.")
        folds = min(folds, len(unique_groups))
        split_iterator = GroupKFold(n_splits=folds).split(
            X_train_orig, y_train, groups_arr
        )
        cv_kind = "group_kfold"
    else:
        split_iterator = StratifiedKFold(
            n_splits=folds, shuffle=True, random_state=random_state
        ).split(X_train_orig, y_train)
        cv_kind = "stratified_kfold"

    classes = np.unique(y_train)
    class_to_index = {label: idx for idx, label in enumerate(classes)}
    base_oof = np.full((len(y_train), len(classes)), np.nan, dtype=float)
    coverage = np.zeros(len(y_train), dtype=int)
    fold_sizes = []
    for fold_index, (fit_idx, hold_idx) in enumerate(split_iterator):
        estimator = clone(mlp_original)
        estimator.fit(X_train_orig[fit_idx], y_train[fit_idx])
        probs = estimator.predict_proba(X_train_orig[hold_idx])
        aligned = np.zeros((len(hold_idx), len(classes)), dtype=float)
        for local_idx, label in enumerate(estimator.classes_):
            aligned[:, class_to_index[label]] = probs[:, local_idx]
        base_oof[hold_idx] = aligned
        coverage[hold_idx] += 1
        fold_sizes.append({
            "fold": fold_index, "fit": int(len(fit_idx)),
            "holdout": int(len(hold_idx)),
        })
    if np.any(coverage != 1) or not np.isfinite(base_oof).all():
        raise RuntimeError("OOF inválido: cada linha deve ser prevista exatamente uma vez.")
    base_test = mlp_original.predict_proba(X_test_orig)
    if base_test.shape[1] != len(classes):
        aligned_test = np.zeros((len(X_test_orig), len(classes)), dtype=float)
        for local_idx, label in enumerate(mlp_original.classes_):
            aligned_test[:, class_to_index[label]] = base_test[:, local_idx]
        base_test = aligned_test
    onto_train = X_train[:, onto_column_indices]
    onto_test = X_test[:, onto_column_indices]
    return (
        np.hstack([base_oof, onto_train]),
        np.hstack([base_test, onto_test]),
        {
            "base_probability_source_train": "out_of_fold",
            "base_probability_source_test": "external_train_fit",
            "cv_folds": folds,
            "cv_kind": cv_kind,
            "coverage_min": int(coverage.min()),
            "coverage_max": int(coverage.max()),
            "fold_sizes": fold_sizes,
        },
    )
