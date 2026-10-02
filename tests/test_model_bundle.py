"""Testes ModelBundle e validação de schema de features."""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.model_bundle import (
    ModelBundle,
    bundle_from_pipeline,
    select_split_for_oracle,
    validate_model_input,
)
from core.mlp_diagnostic import diagnose_oracle


def _pipe(n_in):
    p = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(max_iter=100, random_state=0)),
    ])
    X = np.random.randn(40, n_in)
    y = np.random.randint(0, 2, 40)
    p.fit(X, y)
    return p


def test_validate_model_input_rejects_wrong_width():
    pipe = _pipe(3)
    bundle = bundle_from_pipeline("MLP Original", pipe, ["a", "b", "c"], "original")
    with pytest.raises(ValueError, match="expects 3 features"):
        validate_model_input(bundle, np.random.randn(10, 5))


def test_select_split_for_oracle_original_vs_enriched():
    pipe_orig = _pipe(3)
    bundle_orig = bundle_from_pipeline(
        "MLP Original", pipe_orig, ["a", "b", "c"], "original"
    )
    splits = {
        "original": {
            "X_train": np.zeros((10, 3)),
            "X_test": np.zeros((5, 3)),
        },
        "enriched": {
            "X_train": np.zeros((10, 5)),
            "X_test": np.zeros((5, 5)),
        },
    }
    orig_split = select_split_for_oracle(pipe_orig, splits, bundle=bundle_orig)
    assert orig_split["X_train"].shape[1] == 3

    pipe_onto = _pipe(5)
    bundle_onto = bundle_from_pipeline(
        "MLP_Onto", pipe_onto, ["a", "b", "c", "d", "e"], "enriched"
    )
    enriched_split = select_split_for_oracle(pipe_onto, splits, bundle=bundle_onto)
    assert enriched_split["X_train"].shape[1] == 5


def test_diagnose_oracle_uses_correct_split():
    pipe = _pipe(3)
    bundle = bundle_from_pipeline("MLP Original", pipe, ["a", "b", "c"], "original")
    rng = np.random.default_rng(0)
    splits = {
        "original": {
            "X_train": rng.normal(size=(30, 3)),
            "X_test": rng.normal(size=(10, 3)),
        },
        "enriched": {
            "X_train": rng.normal(size=(30, 5)),
            "X_test": rng.normal(size=(10, 5)),
        },
    }
    diag = diagnose_oracle(pipe, splits=splits, oracle_label="MLP Original", bundle=bundle)
    assert diag["unique_pred_test"] >= 1
