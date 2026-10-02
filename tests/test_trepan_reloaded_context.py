"""Testes de roteamento Trepan-Reloaded por feature_space."""
import sys
from pathlib import Path

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_bundle import bundle_from_pipeline
from core.trepan_reloaded_context import (
    ORACLE_MLP_ORIGINAL,
    ORACLE_MLP_RESIDUAL,
    get_trepan_reloaded_context,
    resolve_reloaded_oracle_key,
)


def _pipe(n):
    p = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=100, random_state=0)),
    ])
    X = np.random.randn(40, n)
    y = np.random.randint(0, 2, 40)
    p.fit(X, y)
    return p


def test_rejected_ontology_routes_original():
    key = resolve_reloaded_oracle_key(
        ORACLE_MLP_RESIDUAL, ontology_enabled=True, ontology_accepted=False
    )
    assert key == "mlp_original"

    orig = _pipe(3)
    split_orig = {
        "X_train": np.random.randn(20, 3),
        "X_test": np.random.randn(10, 3),
        "y_train": np.zeros(20, dtype=int),
        "y_test": np.zeros(10, dtype=int),
    }
    split_enr = {
        "X_train": np.random.randn(20, 5),
        "X_test": np.random.randn(10, 5),
        "y_train": np.zeros(20, dtype=int),
        "y_test": np.zeros(10, dtype=int),
    }
    ctx = get_trepan_reloaded_context(
        ontology_enabled=True,
        ontology_acceptance={"accepted": False},
        selected_oracle_label=ORACLE_MLP_RESIDUAL,
        mlp_original=orig,
        mlp_residual_pipeline=_pipe(5),
        eval_split_original=split_orig,
        eval_split_enriched=split_enr,
        eval_split_residual=split_enr,
        feature_names_original=["a", "b", "c"],
        bundle_original=bundle_from_pipeline(
            ORACLE_MLP_ORIGINAL, orig, ["a", "b", "c"], "original"
        ),
    )
    assert ctx["feature_space"] == "original"
    assert ctx["X_train"].shape[1] == 3
    assert ctx["oracle"] is orig
    assert ctx["mlp_model_onto"] is None


def test_accepted_residual_routes_enriched_matrix_through_wrapper():
    residual = _pipe(5)
    split_res = {
        "X_train": np.random.randn(20, 5),
        "X_test": np.random.randn(10, 5),
        "y_train": np.zeros(20, dtype=int),
        "y_test": np.zeros(10, dtype=int),
    }
    split_enr = {
        "X_train": np.random.randn(20, 7),
        "X_test": np.random.randn(10, 7),
        "y_train": np.zeros(20, dtype=int),
        "y_test": np.zeros(10, dtype=int),
    }
    wrapper = type("ResidualWrapper", (), {
        "n_features_in_": 7,
        "predict": lambda self, X: residual.predict(np.asarray(X)[:, :5]),
    })()
    ctx = get_trepan_reloaded_context(
        ontology_enabled=True,
        ontology_acceptance={"accepted": True},
        selected_oracle_label=ORACLE_MLP_RESIDUAL,
        mlp_original=_pipe(3),
        mlp_residual_pipeline=residual,
        mlp_residual_oracle=wrapper,
        eval_split_original={
            "X_train": np.random.randn(20, 3),
            "X_test": np.random.randn(10, 3),
            "y_train": np.zeros(20, dtype=int),
            "y_test": np.zeros(10, dtype=int),
        },
        eval_split_residual=split_res,
        eval_split_enriched=split_enr,
        feature_names_enriched=[f"e{i}" for i in range(7)],
        feature_names_residual=[f"r{i}" for i in range(5)],
        bundle_residual=bundle_from_pipeline(
            ORACLE_MLP_RESIDUAL, residual, [f"r{i}" for i in range(5)], "residual"
        ),
    )
    assert ctx["feature_space"] == "enriched"
    assert ctx["oracle"] is wrapper
    assert ctx["X_train"].shape[1] == 7
    assert ctx["use_ontology_semantic_pipeline"] is True


def test_original_or_residual_route_forwards_external_cf_samples(monkeypatch):
    from types import SimpleNamespace
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor

    captured = {}

    class CaptureExtractor:
        def __init__(self):
            self.explainer_tree = SimpleNamespace()
            self.last_audit = {'captured': True}

        def extract_tree(self, *args, **kwargs):
            captured.update(kwargs)
            return 'rules'

    monkeypatch.setattr(
        'core.mlp_diagnostic.diagnose_oracle',
        lambda *args, **kwargs: {'ok': True},
    )
    monkeypatch.setattr('core.mlp_diagnostic.log_oracle_diagnostic', lambda *args: None)
    extractor = TrepanReloadedExtractor(ontology=None)
    extractor._original_extractor = CaptureExtractor()
    oracle = _pipe(3)
    X_train = np.random.RandomState(5).randn(20, 3)
    X_validation = np.random.RandomState(6).randn(8, 3)
    extra_X = np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
    extra_y = np.asarray([0, 1])
    extra_weights = np.asarray([2.0, 3.0])
    context = {
        'oracle': oracle,
        'oracle_name': 'MLP Original',
        'feature_space': 'original',
        'feature_names': ['a', 'b', 'c'],
        'X_train': X_train,
        'X_test': X_validation,
        'y_train': np.asarray([0, 1] * 10),
        'y_test': np.asarray([0, 1] * 4),
    }
    result = extractor._extract_tree_in_oracle_space(
        context,
        ['zero', 'one'],
        sample_size=40,
        extra_X=extra_X,
        extra_y=extra_y,
        extra_weights=extra_weights,
    )
    assert result == 'rules'
    assert captured['extra_X'] is extra_X
    assert captured['extra_y'] is extra_y
    assert captured['extra_weights'] is extra_weights
