"""
Regressão: MLP_Onto (oráculo enriquecido) e ONTO_FEATURE_BIAS_WEIGHT configurável.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.mlp_trainer import MLPTrainer
from core.trepan import TrepanReloaded
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from core.metrics_comparator import MetricsComparator


def test_onto_feature_bias_weight_constructor():
    ext = TrepanReloadedExtractor(ontology=None, onto_feature_bias_weight=3.5)
    assert ext.onto_feature_bias_weight == 3.5
    assert ext.ONTO_FEATURE_BIAS_WEIGHT == 3.5


def test_onto_feature_bias_weight_applied_in_gain_weights():
    ext = TrepanReloadedExtractor(ontology=None, onto_feature_bias_weight=4.0)
    names = ["Age", "onto_Age_concept", "Income"]
    ext.feature_semantics = {
        0: {"ontology_mapped": True, "mapping_score": 0.9},
        1: {"ontology_derived": True},
    }
    weights = ext._build_feature_gain_weights(names)
    assert weights[1] == 4.0
    assert weights[0] == pytest.approx(1.2)  # MAPPED_FEATURE_GAIN_BOOST


def test_select_oracle_model_prefers_mlp_onto_when_dimensions_match():
    ext = TrepanReloadedExtractor(ontology=None)
    rng = np.random.default_rng(0)
    X_base = rng.random((30, 5))
    X_aug = np.hstack([X_base, rng.random((30, 3))])

    mlp_orig = MLPClassifier(max_iter=200, random_state=42)
    mlp_orig.fit(X_base, rng.integers(0, 2, size=30))

    mlp_onto = MLPClassifier(max_iter=200, random_state=42)
    mlp_onto.fit(X_aug, rng.integers(0, 2, size=30))

    chosen = ext._select_oracle_model(mlp_orig, mlp_onto, X_aug.shape[1])
    assert chosen is mlp_onto

    chosen_base = ext._select_oracle_model(mlp_orig, mlp_onto, X_base.shape[1])
    assert chosen_base is mlp_orig


def test_resolve_schemas_with_mlp_onto_uses_full_matrix_names():
    ext = TrepanReloadedExtractor(ontology=None)
    rng = np.random.default_rng(1)
    base_names = [f"f{i}" for i in range(4)]
    aug_names = base_names + ["onto_f0_concept", "onto_f1_depth"]
    X = rng.random((20, len(aug_names)))

    mlp_orig = MLPClassifier(max_iter=200, random_state=1)
    mlp_orig.fit(X[:, :4], rng.integers(0, 2, size=20))

    mlp_onto = MLPClassifier(max_iter=200, random_state=1)
    mlp_onto.fit(X, rng.integers(0, 2, size=20))

    matrix_names, base_names_out, oracle_names = ext._resolve_feature_schemas(
        X, aug_names, mlp_orig, base_names, mlp_model_onto=mlp_onto
    )
    assert len(base_names_out) == 4
    assert len(matrix_names) == len(aug_names)
    assert oracle_names == aug_names


def test_train_mlp_onto_sets_oracle_space_metadata():
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
    aug_names = ["f0", "f1", "f2", "onto_a", "onto_b"]

    msg, model = trepan.train_mlp_onto(X_aug, y_enc, augmented_feature_names=aug_names, optimize=False)
    assert model is not None
    assert "MLP_Onto" in msg
    assert trepan.mlp_trainer_onto.arff_meta.get("oracle_space") == "augmented"
    assert model.n_features_in_ == X_aug.shape[1]


def test_fidelity_reference_mlp_onto():
    rng = np.random.default_rng(3)
    X = rng.random((10, 6))
    mlp = MLPClassifier(max_iter=100, random_state=0)
    mlp.fit(X, rng.integers(0, 2, size=10))
    names = [f"c{i}" for i in range(6)]

    assert MetricsComparator._reloaded_fidelity_reference(mlp, names) == "mlp_onto"
    assert MetricsComparator._reloaded_fidelity_reference(mlp, names[:4]) == "mlp_original"
    assert MetricsComparator._reloaded_fidelity_reference(None, names) == "mlp_original"


def test_tree_criterion_defaults_to_entropy():
    ext = TrepanReloadedExtractor(ontology=None)
    assert ext.TREE_CRITERION == "entropy"
    assert ext.CRITERION_SEARCH_ORDER[0] == "entropy"


def test_compare_metrics_reloaded_tree_rejects_unfitted_schema_expansion():
    """Uma matriz ARFF não pode ganhar colunas ontológicas por preenchimento."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(14)
    n_base, n_onto = 14, 30
    base_names = [f"a{i}" for i in range(n_base)]
    full_names = base_names + [f"onto_{i}" for i in range(n_onto)]

    X_base = rng.random((40, n_base))
    X_full = np.hstack([X_base, rng.random((40, n_onto))])
    y = rng.integers(0, 2, size=40)
    X_test = rng.random((12, n_base))
    y_test = rng.integers(0, 2, size=12)

    mlp = MLPClassifier(max_iter=200, random_state=0)
    mlp.fit(X_base, y)

    tree_orig = DecisionTreeClassifier(max_depth=4, random_state=0)
    tree_orig.fit(X_base, y)

    tree_rel = DecisionTreeClassifier(max_depth=4, random_state=1)
    tree_rel.fit(X_full, y)
    assert tree_rel.tree_.n_features == len(full_names)

    ext = TrepanReloadedExtractor(ontology=object())
    ext.ontology_active = True
    ext._base_feature_names = base_names
    ext._matrix_feature_names = full_names
    ext._X_augmented_reference = X_full

    def transform(tree, X):
        return ext._prepare_tree_matrix(tree, X, full_names)

    mc = MetricsComparator(enable_cache=False)
    with pytest.raises(ValueError, match="fitted|schema|transform",):
        mc.compare_all_models(
            mlp, tree_orig, tree_rel,
            X_test, y_test,
            feature_names=base_names,
            feature_names_reloaded=full_names,
            reload_test_transform=transform,
            ontology_active=True,
        )


def test_set_onto_feature_bias_weight_invalidates_multipliers():
    ext = TrepanReloadedExtractor(ontology=None, onto_feature_bias_weight=2.5)
    names = ["f0", "onto_f0_concept"]
    ext._build_ontology_split_multipliers(names)
    assert ext._ontology_split_multipliers is not None
    ext.set_onto_feature_bias_weight(4.0)
    assert ext.onto_feature_bias_weight == 4.0
    assert ext._ontology_split_multipliers is None


def test_synthetic_sampling_uses_original_mlp_not_onto():
    """Oráculo dual: amostragem ARFF não deve chamar predict_proba do MLP_Onto."""
    rng = np.random.default_rng(42)
    n_base, n_extra = 5, 3
    X_base = rng.random((20, n_base))

    mlp_orig = MLPClassifier(max_iter=200, random_state=0)
    mlp_orig.fit(X_base, rng.integers(0, 2, size=20))

    X_aug = np.hstack([X_base, rng.random((20, n_extra))])
    mlp_onto = MLPClassifier(max_iter=200, random_state=1)
    mlp_onto.fit(X_aug, rng.integers(0, 2, size=20))

    ext = TrepanReloadedExtractor(ontology=object())
    ext.ontology_active = True
    ext._mlp_model_original = mlp_orig
    ext._mlp_model_onto = mlp_onto
    ext.feature_centrality = {i: 0.5 for i in range(n_base)}

    onto_calls = []

    def onto_proba_trap(X):
        onto_calls.append(X.shape[1])
        raise AssertionError("MLP_Onto não deve ser usado na amostragem ARFF")

    mlp_onto.predict_proba = onto_proba_trap
    mlp_onto.predict = lambda X: (_ for _ in ()).throw(
        AssertionError("MLP_Onto não deve ser usado na amostragem ARFF")
    )

    result = ext._generate_ontology_aware_synthetic_data(
        mlp_orig, X_base, sample_size=30, feature_names=[f"f{i}" for i in range(n_base)]
    )
    assert result.shape[1] == n_base
    assert len(onto_calls) == 0
