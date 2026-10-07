"""Registo do pré-processamento APRENDIDO de uma unidade experimental (split): tudo é ajustado só no treino.

O benchmark não aplica imputação/encoding/seleção/normalização fora dos estimadores: o dataset já chega como matriz numérica
finita e a normalização do MLP vive dentro do pipeline ajustado no treino. O ``preprocessing_id`` identifica a representação
base de treino partilhada por C, D, E e F (a única diferença entre D/E/F é a componente semântica).
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Sequence

import numpy as np

from core.benchmark.manifest import stable_hash


def preprocessing_record(X_train, feature_names: Sequence[str], train_idx) -> Dict[str, Any]:
    X = np.ascontiguousarray(np.asarray(X_train, dtype=float))
    train_hash = hashlib.sha256(X.tobytes()).hexdigest()
    record = {
        "representation": "numeric_matrix_as_loaded",
        "imputation": "none", "encoding": "none", "feature_selection": "none",
        "scaling": "estimator_internal_pipeline_fit_on_train_only",
        "semantic_enrichment": "provider_fit_on_train_only",
        "fit_partition": "train", "test_used_in_fit": False,
        "n_fit_rows": int(len(X)), "n_features": int(X.shape[1]),
        "feature_names_hash": stable_hash(list(map(str, feature_names))),
        "train_index_hash": stable_hash([int(i) for i in np.asarray(train_idx)]),
        "train_matrix_hash": train_hash,
    }
    # Identidade CIENTÍFICA: só valores/partição/configuração (nomes das features não alteram nenhum resultado).
    scientific = {k: v for k, v in record.items() if k != "feature_names_hash"}
    record["scientific_preprocessing_id"] = stable_hash(scientific)[:16]
    # Identidade de ARTEFACTO/cache: acrescenta os nomes das features (rastreabilidade do esquema). Nunca seleciona comportamento.
    record["preprocessing_id"] = stable_hash({k: v for k, v in record.items() if k != "scientific_preprocessing_id"})[:16]
    return record
