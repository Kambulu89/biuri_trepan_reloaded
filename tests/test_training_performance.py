"""Testes para presets de treino e seleção de features ontológicas."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.training_config import get_training_preset, TRAINING_PRESETS, grid_param_grid_for_preset
from core.onto_feature_selector import select_top_ontology_features
from core.model_cache import compute_dataset_hash, build_cache_key


def test_training_presets_exist():
    assert set(TRAINING_PRESETS.keys()) == {'fast', 'balanced', 'scientific'}
    fast = get_training_preset('fast')
    assert fast.optuna_trials == 10
    assert fast.run_grid is False
    assert fast.top_k_onto_features == 30
    sci = get_training_preset('scientific')
    assert sci.optuna_trials == 50
    assert sci.top_k_onto_features is None


def test_grid_for_fast_preset():
    fast = get_training_preset('fast')
    assert grid_param_grid_for_preset(fast) is None or fast.run_grid is False


def test_onto_feature_selection_keeps_originals():
    rng = np.random.default_rng(0)
    base = ['f0', 'f1']
    onto = [f'onto_extra_{i}' for i in range(20)]
    names = base + onto
    X = rng.normal(size=(100, len(names)))
    y = (X[:, 0] > 0).astype(int)
    X_out, names_out, info = select_top_ontology_features(
        X, y, names, base, top_k=5
    )
    assert all(n in names_out for n in base)
    assert len(names_out) == len(base) + 5
    assert info['selected_onto'] == 5


def test_cache_key_stable():
    X = np.random.randn(50, 4)
    y = np.random.randint(0, 2, 50)
    h1 = compute_dataset_hash(X, y, 'test')
    h2 = compute_dataset_hash(X, y, 'test')
    assert h1 == h2
    preset = get_training_preset('balanced')
    key = build_cache_key(
        model_role='mlp_original',
        dataset_hash=h1,
        ontology_hash='no_ontology',
        preprocessing_hash='abc',
        preset=preset,
    )
    assert len(key) == 64


def test_dataset_hash_with_categorical_strings():
    X = np.array([
        [39, 'Private'],
        [50, 'Self-emp-not-inc'],
        [28, 'Local-gov'],
    ], dtype=object)
    y = np.array(['<=50K', '>50K', '<=50K'], dtype=object)
    h = compute_dataset_hash(X, y, 'adult')
    assert isinstance(h, str) and len(h) == 16


def test_mlp_trainer_encodes_numpy_object_array():
    from core.mlp_trainer import MLPTrainer

    trainer = MLPTrainer()
    X = np.array([
        [39, 'Private'],
        [50, 'Self-emp-not-inc'],
    ], dtype=object)
    y = np.array(['<=50K', '>50K'], dtype=object)
    trainer._fit_encoders(X)
    X_enc = trainer._encode_features(X)
    y_enc = trainer.label_encoder.fit_transform(y)
    assert X_enc.shape == (2, 2)
    assert y_enc.shape == (2,)
