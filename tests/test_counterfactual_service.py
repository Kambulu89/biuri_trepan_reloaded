"""Testes expandidos do subsistema de contrafactuais."""
import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

from counterfactuals._bootstrap import setup

setup()

from counterfactuals.improve_surrogate import improve_surrogate, evaluate_improvement
from counterfactuals.dataset_config import get_dataset_config, build_config_from_arff_meta, ALL_DATASETS
from counterfactuals.service import (
    evaluate_consistency,
    generate_counterfactuals_from_session,
    _select_instances_stratified,
)
from core.trepan_original import TrepanOriginalExtractor


class _Oracle:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(np.asarray(X, dtype=float))

    def predict_proba(self, X):
        return self.model.predict_proba(np.asarray(X, dtype=float))


@pytest.fixture
def toy_setup():
    rng = np.random.RandomState(0)
    X = rng.randn(80, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(8,), max_iter=500, random_state=0)
    mlp.fit(X, y)

    extractor = TrepanOriginalExtractor()
    extractor.extract_tree(
        mlp, X, y, sample_size=200,
        feature_names=[f"f{i}" for i in range(4)],
        class_names=["0", "1"],
    )
    original = extractor.explainer_tree

    cfs = []
    for i in range(5):
        cf = X[i].copy()
        cf[0] *= -1
        cfs.append({"cf": cf, "original_class": int(y[i])})

    return mlp, X, y, extractor, original, cfs


def test_improve_surrogate_core_extractor(toy_setup):
    mlp, X, y, extractor, original, cfs = toy_setup
    extractor.explainer_tree = original
    improved, stats = improve_surrogate(
        mlp_model=mlp,
        X_train=X,
        y_train=y,
        tree_extractor=extractor,
        cf_list=cfs,
        feature_names=[f"f{i}" for i in range(4)],
        class_names=["0", "1"],
        max_cfs=3,
        seed=0,
    )
    assert improved is not None
    assert stats["total_cfs"] == 5


def test_cogs_smoke_toy():
    from counterfactuals.pipelines.pipeline_counterfactuals import run_cogs_for_single_instance

    rng = np.random.RandomState(1)
    X = rng.randn(20, 3)
    y = (X[:, 0] > 0).astype(int)
    mlp = MLPClassifier(hidden_layer_sizes=(6,), max_iter=300, random_state=1)
    mlp.fit(X, y)
    oracle = _Oracle(mlp)
    intervals = np.array([(-3.0, 3.0)] * 3, dtype=object)
    args = (0, X[0], int(y[0]), oracle, intervals, [], {0: '0', 1: '1'}, False)
    result = run_cogs_for_single_instance(args)
    assert result is None or isinstance(result, (dict, list))


def test_select_instances_stratified():
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    idx = _select_instances_stratified(y, fraction=0.33, seed=0)
    assert len(idx) >= 3
    assert len(idx) <= len(y)


def test_generate_counterfactuals_from_session_cogs_only(toy_setup, monkeypatch):
    mlp, X, y, extractor, original, cfs = toy_setup
    reloaded = DecisionTreeClassifier(max_depth=3, random_state=0)
    reloaded.fit(X, y)

    session = {
        'mlp_oracle': _Oracle(mlp),
        'X_train_enc': X,
        'y_train_enc': y,
        'tree_a': original,
        'tree_b': reloaded,
        'transformed_feature_names': [f"f{i}" for i in range(4)],
        'class_labels': {0: '0', 1: '1'},
        'is_multiclass': False,
        'config': {'clear_max_predictors': 1, 'class_labels': {0: '0', 1: '1'}, 'is_multiclass': False},
    }

    def fake_clear(*args, **kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        'counterfactuals.pipelines.pipeline_counterfactuals.run_clear_cfs',
        fake_clear,
    )

    result = generate_counterfactuals_from_session(session)
    assert 'consistency' in result
    assert len(result['consistency']) == 4


def test_dataset_config_all_present():
    for name in ALL_DATASETS:
        cfg = get_dataset_config(name)
        assert 'file' in cfg
        assert cfg['csv_file'].endswith('.csv')


def test_build_config_from_arff_meta():
    meta = {'features': ['a', 'b'], 'classes': ['x', 'y'], 'target': 'class', 'file_name': 't.arff'}
    cfg = build_config_from_arff_meta(meta)
    assert cfg['is_multiclass'] is False
    assert cfg['class_labels'] == {0: 'x', 1: 'y'}


def test_mlp_trainer_bypass():
    from core.mlp_trainer import MLPTrainer

    rng = np.random.RandomState(0)
    X_raw = rng.randn(40, 3)
    y_raw = (X_raw[:, 0] > 0).astype(int)
    trainer = MLPTrainer()
    trainer.train(X_raw.tolist(), y_raw.tolist(), optimize=False)
    X_enc = trainer.transform(X_raw)
    trainer.bypass_preprocessing = True
    pred1 = trainer.predict(X_enc)
    pred2 = trainer.predict(X_raw)
    assert np.array_equal(pred1, pred2)


def test_service_evaluate_consistency_missing():
    assert evaluate_consistency('nonexistent_dataset_xyz') is None
