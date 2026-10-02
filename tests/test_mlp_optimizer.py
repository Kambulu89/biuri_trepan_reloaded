"""
Testes do módulo de otimização MLP.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mlp_optimizer import (
    OPTUNA_AVAILABLE,
    optimize_mlp_with_grid_search,
    run_full_mlp_optimization,
    train_baseline_mlp,
    _choose_scoring,
)
from core.mlp_trainer import MLPTrainer
from core.trepan import TrepanReloaded


@pytest.fixture
def small_split():
    X, y = make_classification(
        n_samples=120,
        n_features=8,
        n_informative=6,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    split = int(0.7 * len(y))
    return X[:split], y[:split], X[split:], y[split:]


def test_choose_scoring_balanced_vs_imbalanced():
    balanced = np.array([0, 0, 1, 1, 2, 2])
    assert _choose_scoring(balanced) == "accuracy"
    imbalanced = np.array([0] * 90 + [1] * 10)
    assert _choose_scoring(imbalanced) == "balanced_accuracy"


def test_baseline_mlp_pipeline_with_scaler(small_split):
    X_train, y_train, X_test, y_test = small_split
    result = train_baseline_mlp(
        X_train, y_train, X_test, y_test, "MLP Original", n_features=X_train.shape[1]
    )
    assert result["model"] is not None
    assert "scaler" in result["model"].named_steps
    assert result["test_metrics"]["accuracy"] >= 0.0


def test_grid_search_runs_on_small_data(small_split):
    X_train, y_train, X_test, y_test = small_split
    grid = {
        "mlp__hidden_layer_sizes": [(32,), (16,)],
        "mlp__activation": ["relu"],
        "mlp__solver": ["adam"],
        "mlp__alpha": [0.001],
        "mlp__learning_rate_init": [0.001],
        "mlp__max_iter": [300],
        "mlp__early_stopping": [True],
        "mlp__validation_fraction": [0.1],
    }
    import core.mlp_optimizer as mo

    original_grid = mo.MLP_GRID_PARAM_GRID
    mo.MLP_GRID_PARAM_GRID = grid
    try:
        result = optimize_mlp_with_grid_search(
            X_train, y_train, X_test, y_test, "MLP Original"
        )
    finally:
        mo.MLP_GRID_PARAM_GRID = original_grid

    assert result["optimization_method"] == "grid_search"
    assert result["best_params"]
    assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0


@pytest.mark.skipif(not OPTUNA_AVAILABLE, reason="Optuna não instalado")
def test_run_full_mlp_optimization_selects_best(small_split):
    X_train, y_train, X_test, y_test = small_split
    best, all_results = run_full_mlp_optimization(
        X_train,
        y_train,
        X_test,
        y_test,
        model_name="MLP Original",
        dataset_name="test_dataset",
        run_grid=False,
        run_optuna=False,
    )
    assert best["model"] is not None
    assert len(all_results) == 1
    assert best["optimization_method"] == "baseline"


def test_mlp_trainer_optimize_false_preserves_api():
    rng = np.random.default_rng(0)
    X_raw = rng.random((30, 4)).tolist()
    y = ["A" if i % 2 == 0 else "B" for i in range(30)]
    trainer = MLPTrainer()
    model, X_enc, y_enc = trainer.train(X_raw, y, optimize=False)
    assert model is not None
    assert X_enc.shape == (30, 4)
    assert len(y_enc) == 30


def test_trepan_train_mlp_onto_without_optimization():
    trepan = TrepanReloaded(use_default_ontology=False)
    rng = np.random.default_rng(2)
    X_raw = [[str(v) for v in row] for row in rng.integers(0, 3, size=(25, 3))]
    y = ["A" if i % 2 == 0 else "B" for i in range(25)]
    trepan.train_mlp(X_raw, y, optimize=False)

    X_aug = np.hstack([
        trepan.mlp_trainer._encode_features(X_raw),
        rng.random((25, 2)),
    ])
    y_enc = trepan.label_encoder.transform(y)
    msg, model = trepan.train_mlp_onto(
        X_aug, y_enc, augmented_feature_names=["f0", "f1", "f2", "onto_a", "onto_b"],
        optimize=False,
    )
    assert model is not None
    assert "MLP_Onto" in msg
