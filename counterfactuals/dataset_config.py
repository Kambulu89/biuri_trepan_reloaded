"""Configuração de datasets e resolução de caminhos para contrafactuais."""
from __future__ import annotations

import copy
from typing import Any, Dict, Optional


def build_config_from_arff_meta(
    meta: Dict[str, Any],
    numeric_features: Optional[list] = None,
    categorical_features: Optional[list] = None,
) -> Dict[str, Any]:
    """Constrói config CF a partir de metadados ARFF (GUI)."""
    features = meta.get('features') or []
    classes = meta.get('classes') or []
    numeric = numeric_features if numeric_features is not None else features
    categorical = categorical_features or []
    class_labels = {i: str(c) for i, c in enumerate(classes)}
    config = {
        'csv_file': meta.get('file_name', 'arff_dataset'),
        'target': meta.get('target', 'class'),
        'numeric_features': list(numeric),
        'categorical_features': list(categorical),
        'class_labels': class_labels,
        'is_multiclass': len(classes) > 2,
        'mlp_params': {},
        'tune_mlp': False,
        'clear_max_predictors': 2,
    }
    # Metadados opcionais: mantêm as restrições fora do código do gerador e
    # permitem que datasets clínicos/financeiros declarem regras próprias.
    for key in (
        'feature_ranges', 'immutable_features', 'actionable_features',
        'protected_features', 'categorical_values', 'binary_features',
        'causal_rules',
    ):
        if meta.get(key) is not None:
            config[key] = copy.deepcopy(meta[key])
    return config
