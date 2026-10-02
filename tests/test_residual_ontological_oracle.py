"""
Regressão: MLP Residual Ontológico e critério de aceitação ontológica.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.residual_ontological_oracle import (
    ResidualOntologicalOracle,
    build_residual_feature_matrix,
    onto_column_indices,
    original_column_indices,
)
from core.ontology_acceptance import (
    evaluate_ontology_acceptance,
    select_oracle_model,
)
from core.trepan import TrepanReloaded
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.metrics_comparator import MetricsComparator


def _make_dual_data(n_samples=40, n_base=4, n_onto=2, seed=0):
    rng = np.random.default_rng(seed)
    base_names = [f"f{i}" for i in range(n_base)]
    onto_names = [f"onto_f{i}_concept" for i in range(n_onto)]
    aug_names = base_names + onto_names
    X_base = rng.random((n_samples, n_base))
    X_onto = rng.random((n_samples, n_onto))
    X_aug = np.hstack([X_base, X_onto])
    y = rng.integers(0, 2, size=n_samples)
    return X_base, X_aug, y, base_names, aug_names


def test_build_residual_feature_matrix_shape():
    X_base, X_aug, y, base_names, aug_names = _make_dual_data()
    mlp_orig = MLPClassifier(max_iter=300, random_state=0)
    mlp_orig.fit(X_base, y)
    orig_idx = original_column_indices(aug_names, base_names)
    onto_idx = onto_column_indices(aug_names, base_names)
    X_res = build_residual_feature_matrix(X_aug, mlp_orig, orig_idx, onto_idx)
    assert X_res.shape[0] == len(y)
    assert X_res.shape[1] == len(mlp_orig.classes_) + len(onto_idx)


def test_residual_oracle_predict_on_enriched_matrix():
    X_base, X_aug, y, base_names, aug_names = _make_dual_data()
    mlp_orig = MLPClassifier(max_iter=300, random_state=0)
    mlp_orig.fit(X_base, y)
    orig_idx = original_column_indices(aug_names, base_names)
    onto_idx = onto_column_indices(aug_names, base_names)
    X_res = build_residual_feature_matrix(X_aug, mlp_orig, orig_idx, onto_idx)
    mlp_res = MLPClassifier(max_iter=300, random_state=1)
    mlp_res.fit(X_res, y)
    oracle = ResidualOntologicalOracle(
        mlp_orig, mlp_res, orig_idx, onto_idx, X_aug.shape[1]
    )
    assert oracle.n_features_in_ == X_aug.shape[1]
    preds = oracle.predict(X_aug)
    assert preds.shape == (len(y),)


def test_evaluate_ontology_acceptance_strict_noninferiority():
    accepted = evaluate_ontology_acceptance(0.75, 0.74, 0.70, 0.72, tolerance=0.01)
    assert accepted["accepted"] is True  # ambas dentro da margem
    rejected = evaluate_ontology_acceptance(0.75, 0.73, 0.70, 0.68, tolerance=0.01)
    assert rejected["accepted"] is False
    assert rejected["ontology_impact"] == "harmful"
    one_metric_only = evaluate_ontology_acceptance(
        0.80, 0.70, 0.80, 0.82,
        balanced_accuracy_original=0.80,
        balanced_accuracy_onto=0.82,
        tolerance=0.01,
    )
    assert one_metric_only["accepted"] is False


def test_select_oracle_model():
    X_base, X_aug, y, base_names, aug_names = _make_dual_data()
    mlp_orig = MLPClassifier(max_iter=200, random_state=0)
    mlp_orig.fit(X_base, y)
    orig_idx = original_column_indices(aug_names, base_names)
    onto_idx = onto_column_indices(aug_names, base_names)
    X_res = build_residual_feature_matrix(X_aug, mlp_orig, orig_idx, onto_idx)
    mlp_res = MLPClassifier(max_iter=200, random_state=1)
    mlp_res.fit(X_res, y)
    oracle = ResidualOntologicalOracle(
        mlp_orig, mlp_res, orig_idx, onto_idx, X_aug.shape[1]
    )
    acc = evaluate_ontology_acceptance(0.5, 0.6, 0.5, 0.6)
    chosen, label = select_oracle_model(mlp_orig, oracle, acc)
    assert chosen is oracle
    assert label == "MLP Residual Ontológico"
    bad = evaluate_ontology_acceptance(0.9, 0.5, 0.9, 0.5)
    chosen2, label2 = select_oracle_model(mlp_orig, oracle, bad)
    assert chosen2 is mlp_orig
    assert label2 == "MLP Original"


def test_extractor_selects_residual_oracle_when_dims_match():
    X_base, X_aug, y, base_names, aug_names = _make_dual_data()
    mlp_orig = MLPClassifier(max_iter=200, random_state=0)
    mlp_orig.fit(X_base, y)
    orig_idx = original_column_indices(aug_names, base_names)
    onto_idx = onto_column_indices(aug_names, base_names)
    X_res = build_residual_feature_matrix(X_aug, mlp_orig, orig_idx, onto_idx)
    mlp_res = MLPClassifier(max_iter=200, random_state=1)
    mlp_res.fit(X_res, y)
    oracle = ResidualOntologicalOracle(
        mlp_orig, mlp_res, orig_idx, onto_idx, X_aug.shape[1]
    )
    ext = TrepanReloadedExtractor(ontology=None)
    chosen = ext._select_oracle_model(mlp_orig, oracle, X_aug.shape[1])
    assert chosen is oracle


def test_train_mlp_residual_onto_integration():
    trepan = TrepanReloaded(use_default_ontology=False)
    rng = np.random.default_rng(5)
    X_raw = [[str(v) for v in row] for row in rng.integers(0, 3, size=(30, 3))]
    y = ["A" if i % 2 == 0 else "B" for i in range(30)]
    trepan.train_mlp(X_raw, y, optimize=False)
    base_names = ["f0", "f1", "f2"]
    aug_names = base_names + ["onto_a", "onto_b"]
    X_aug = np.hstack([
        trepan.mlp_trainer._encode_features(X_raw),
        rng.random((30, 2)),
    ])
    y_enc = trepan.label_encoder.transform(y)
    msg, oracle, acceptance, selected = trepan.train_mlp_residual_onto(
        X_aug,
        y_enc,
        trepan.mlp_model,
        base_names,
        aug_names,
        optimize=False,
    )
    assert oracle is not None
    assert acceptance is not None
    assert selected is not None
    assert "MLP Residual" in msg or "Oráculo" in msg


def test_fidelity_reference_residual_oracle():
    X_base, X_aug, y, base_names, aug_names = _make_dual_data(n_samples=20)
    mlp_orig = MLPClassifier(max_iter=200, random_state=0)
    mlp_orig.fit(X_base, y)
    orig_idx = original_column_indices(aug_names, base_names)
    onto_idx = onto_column_indices(aug_names, base_names)
    X_res = build_residual_feature_matrix(X_aug, mlp_orig, orig_idx, onto_idx)
    mlp_res = MLPClassifier(max_iter=200, random_state=1)
    mlp_res.fit(X_res, y)
    oracle = ResidualOntologicalOracle(
        mlp_orig, mlp_res, orig_idx, onto_idx, X_aug.shape[1]
    )
    ref = MetricsComparator._reloaded_fidelity_reference(oracle, aug_names)
    assert ref == "mlp_onto"
