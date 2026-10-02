"""Configuração de datasets e resolução de caminhos para contrafactuais."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

from counterfactuals._paths import DATASETS_DIR

ALL_DATASETS = ['iris', 'wine', 'german_credit', 'wdbc', 'sonar', 'hepatitis']

DATASET_CONFIGS: Dict[str, Dict[str, Any]] = {
    'iris': {
        'csv_file': 'iris.csv',
        'target': 'class',
        'numeric_features': ['sepal_length', 'sepal_width', 'petal_length', 'petal_width'],
        'categorical_features': [],
        'class_labels': {0: 'Iris-setosa', 1: 'Iris-versicolor', 2: 'Iris-virginica'},
        'is_multiclass': True,
        'mlp_params': {
            'hidden_layer_sizes': (50, 30), 'max_iter': 1000,
            'learning_rate_init': 0.01, 'early_stopping': False, 'random_state': 42,
        },
        'tune_mlp': False,
        'clear_max_predictors': 1,
    },
    'wine': {
        'csv_file': 'wine.csv',
        'target': 'class',
        'numeric_features': [
            'Alcohol', 'Malic_acid', 'Ash', 'Alcalinity_of_ash', 'Magnesium',
            'Total_phenols', 'Flavanoids', 'Nonflavanoid_phenols', 'Proanthocyanins',
            'Color_intensity', 'Hue', 'OD280_OD315_of_diluted_wines', 'Proline',
        ],
        'categorical_features': [],
        'class_labels': {0: 'class_1', 1: 'class_2', 2: 'class_3'},
        'is_multiclass': True,
        'mlp_params': {
            'hidden_layer_sizes': (50, 30), 'max_iter': 1000,
            'learning_rate_init': 0.01, 'early_stopping': False, 'random_state': 42,
        },
        'tune_mlp': False,
        'clear_num_samples': 800,
        'clear_regression_sample_size': 80,
        'clear_max_predictors': 1,
    },
    'german_credit': {
        'csv_file': 'german_credit.csv',
        'target': 'class',
        'numeric_features': ['attr_2', 'attr_5', 'attr_8', 'attr_11', 'attr_13', 'attr_16', 'attr_18'],
        'categorical_features': [
            'attr_1', 'attr_3', 'attr_4', 'attr_6', 'attr_7', 'attr_9', 'attr_10',
            'attr_12', 'attr_14', 'attr_15', 'attr_17', 'attr_19', 'attr_20',
        ],
        'class_labels': {0: 'pay', 1: 'default'},
        'is_multiclass': False,
        'mlp_params': {
            'hidden_layer_sizes': (50,), 'alpha': 0.05, 'learning_rate_init': 0.001,
            'max_iter': 5000, 'early_stopping': True, 'validation_fraction': 0.15,
            'n_iter_no_change': 60, 'random_state': 42,
        },
        'balance_method': 'oversample',
        'tune_mlp': True,
        'param_grid': {
            'classifier__hidden_layer_sizes': [(50,), (100,), (100, 50), (200, 100)],
            'classifier__alpha': [0.001, 0.01, 0.05, 0.1],
            'classifier__learning_rate_init': [0.001, 0.01],
        },
        'tree_sample_size': 800,
        'clear_num_samples': 600,
        'clear_regression_sample_size': 60,
        'clear_max_predictors': 2,
    },
    'wdbc': {
        'csv_file': 'wdbc.csv',
        'target': 'diagnosis',
        'numeric_features': [f'f_{i}' for i in range(1, 31)],
        'categorical_features': [],
        'class_labels': {0: 'Benign', 1: 'Malignant'},
        'is_multiclass': False,
        'mlp_params': {'hidden_layer_sizes': (50, 30), 'max_iter': 1000, 'random_state': 42},
        'tune_mlp': False,
        'clear_num_samples': 800,
        'clear_regression_sample_size': 80,
        'clear_max_predictors': 2,
    },
    'sonar': {
        'csv_file': 'sonar.csv',
        'target': 'class',
        'numeric_features': [f'attribute_{i}' for i in range(1, 61)],
        'categorical_features': [],
        'class_labels': {0: 'Rock', 1: 'Mine'},
        'is_multiclass': False,
        'mlp_params': {
            'hidden_layer_sizes': (30, 15), 'alpha': 0.1, 'learning_rate_init': 0.01,
            'max_iter': 2000, 'random_state': 42,
        },
        'tune_mlp': False,
        'tree_sample_size': 600,
        'clear_num_samples': 600,
        'clear_regression_sample_size': 60,
        'clear_max_predictors': 1,
    },
    'hepatitis': {
        'csv_file': 'hepatitis.csv',
        'target': 'class',
        'numeric_features': [f'feature_{i}' for i in range(1, 20)],
        'categorical_features': [],
        'class_labels': {0: 'DIE', 1: 'LIVE'},
        'is_multiclass': False,
        'mlp_params': {
            'hidden_layer_sizes': (6,), 'activation': 'logistic', 'alpha': 0.1,
            'learning_rate_init': 0.001, 'max_iter': 5000, 'early_stopping': False,
            'random_state': 42,
        },
        'balance_method': 'oversample',
        'tune_mlp': False,
        'use_minmax_scaler': True,
        'calibrate_threshold': True,
        'clear_num_samples': 800,
        'clear_regression_sample_size': 80,
        'clear_max_predictors': 1,
    },
}

MLP_ACCURACY = {
    'iris': 0.933, 'wine': 0.981, 'german_credit': 0.757,
    'wdbc': 0.953, 'sonar': 0.873, 'hepatitis': 0.702,
}


def get_dataset_config(name: str) -> Dict[str, Any]:
    if name not in DATASET_CONFIGS:
        raise KeyError(f"Dataset desconhecido: {name}. Opções: {ALL_DATASETS}")
    cfg = copy.deepcopy(DATASET_CONFIGS[name])
    cfg['file'] = str(DATASETS_DIR / cfg['csv_file'])
    return cfg


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
