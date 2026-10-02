"""
Testes de detecção de oráculo MLP degenerado.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mlp_diagnostic import (
    diagnose_mlp_original,
    ensure_robust_mlp_original,
    log_mlp_original_diagnostic,
)
from core.mlp_optimizer import train_robust_mlp_original


@pytest.fixture
def split():
    X, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=8,
        n_redundant=0,
        n_classes=3,
        random_state=42,
    )
    return train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)


def _constant_pipeline(y_train):
    """Pipeline que prevê sempre a classe majoritária."""
    majority = int(np.bincount(y_train).argmax())

    class _ConstantPredictor:
        def fit(self, X, y):
            return self

        def predict(self, X):
            return np.full(len(X), majority)

        n_features_in_ = None

    return Pipeline([("scaler", StandardScaler()), ("mlp", _ConstantPredictor())])


def test_diagnose_detects_degenerate_mlp(split):
    X_train, X_test, y_train, y_test = split
    model = _constant_pipeline(y_train)
    model.fit(X_train, y_train)
    model.named_steps["mlp"].n_features_in_ = X_train.shape[1]

    diag = diagnose_mlp_original(model, X_train, y_train, X_test, y_test)
    assert diag["degenerate"] is True
    assert diag["unique_y_pred_test"] == 1
    assert diag["weak_vs_dummy"] is True


def test_diagnose_healthy_mlp(split):
    X_train, X_test, y_train, y_test = split
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(50,),
            max_iter=500,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    diag = diagnose_mlp_original(model, X_train, y_train, X_test, y_test)
    assert diag["unique_y_pred_test"] >= 2
    assert diag["degenerate"] is False


def test_train_robust_mlp_beats_constant_oracle(split, capsys):
    X_train, X_test, y_train, y_test = split
    result = train_robust_mlp_original(
        X_train, y_train, X_test, y_test, mode="balanced"
    )
    model = result["model"]
    y_pred = model.predict(X_test)
    assert len(np.unique(y_pred)) >= 2
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)
    from sklearn.metrics import f1_score
    mlp_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    dummy_f1 = f1_score(y_test, dummy.predict(X_test), average="weighted", zero_division=0)
    assert mlp_f1 >= dummy_f1 - 0.05


def test_ensure_robust_replaces_degenerate(split):
    X_train, X_test, y_train, y_test = split
    bad = _constant_pipeline(y_train)
    bad.fit(X_train, y_train)
    bad.named_steps["mlp"].n_features_in_ = X_train.shape[1]

    model, diag = ensure_robust_mlp_original(
        X_train, y_train, X_test, y_test, existing_model=bad
    )
    assert diag.get("robust_retrained") is True
    assert len(np.unique(model.predict(X_test))) >= 2


def test_log_mlp_diagnostic_prints_block(split, capsys):
    X_train, X_test, y_train, y_test = split
    model = _constant_pipeline(y_train)
    model.fit(X_train, y_train)
    diag = diagnose_mlp_original(model, X_train, y_train, X_test, y_test)
    log_mlp_original_diagnostic(diag)
    captured = capsys.readouterr()
    assert "[MLP ORIGINAL DIAGNOSTIC]" in captured.out
    assert "degenerado" in captured.out.lower() or "WARNING" in captured.out
