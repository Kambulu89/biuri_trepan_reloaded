"""
Pipeline agnóstico ao número de features: oráculo dual + alinhamento por nome.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.trepan_reloaded_extractor import TrepanReloadedExtractor


@pytest.mark.parametrize("n_base,n_extra", [(3, 2), (12, 5), (31, 8), (50, 10)])
def test_dual_oracle_sampling_and_labeling_dimensions(n_base, n_extra):
    rng = np.random.default_rng(n_base + n_extra)
    base_names = [f"Attr_{i:02d}" for i in range(n_base)]
    aug_names = base_names + [f"onto_extra_{i}" for i in range(n_extra)]
    n_aug = len(aug_names)

    X_base = rng.random((40, n_base))
    X_aug = np.hstack([X_base, rng.random((40, n_extra))])

    mlp_orig = MLPClassifier(max_iter=300, random_state=0)
    mlp_orig.fit(X_base, rng.integers(0, 2, size=40))

    mlp_onto = MLPClassifier(max_iter=300, random_state=1)
    mlp_onto.fit(X_aug, rng.integers(0, 2, size=40))

    ext = TrepanReloadedExtractor(ontology=object())
    ext.ontology_active = True
    ext._mlp_model_original = mlp_orig
    ext._mlp_model_onto = mlp_onto
    ext._original_feature_names = base_names
    ext._base_feature_names = base_names
    ext._matrix_feature_names = aug_names
    ext._oracle_feature_names = aug_names
    ext.feature_centrality = {i: 0.5 for i in range(n_aug)}

    mlp_onto.predict_proba = lambda X: (_ for _ in ()).throw(
        AssertionError("MLP_Onto não deve ser usado na amostragem ARFF")
    )

    X_syn_base = ext._generate_ontology_aware_synthetic_data(
        mlp_orig, X_base, sample_size=50, feature_names=base_names
    )
    assert X_syn_base.shape[1] == n_base

    X_syn = np.hstack([X_syn_base, rng.random((len(X_syn_base), n_extra))])
    y_syn = ext._mlp_predict(mlp_onto, X_syn, aug_names)
    assert len(y_syn) == len(X_syn_base)
    assert mlp_onto.n_features_in_ == n_aug


def test_pick_oracle_columns_rejects_positional_mismatch():
    rng = np.random.default_rng(0)
    n_base = 7
    names = [f"f{i}" for i in range(n_base)]
    X = rng.random((10, n_base))
    mlp = MLPClassifier(max_iter=100, random_state=0)
    mlp.fit(X, rng.integers(0, 2, size=10))

    ext = TrepanReloadedExtractor(ontology=None)
    ext._base_feature_names = names
    ext._oracle_feature_names = names

    indices = ext._pick_oracle_column_indices(names, mlp, oracle_names=names)
    assert len(indices) == n_base

    wide = np.hstack([rng.random((10, 3)), X])
    wide_names = [f"noise_{i}" for i in range(3)] + names
    idx2 = ext._pick_oracle_column_indices(wide_names, mlp, base_names=names)
    assert len(idx2) == n_base
    assert np.allclose(wide[:, idx2], X)


def test_validate_oracle_width_raises_on_mismatch():
    ext = TrepanReloadedExtractor(ontology=None)
    mlp = MLPClassifier(max_iter=50, random_state=0)
    mlp.fit(np.random.random((5, 4)), [0, 1, 0, 1, 0])
    with pytest.raises(ValueError, match="espera 4 features"):
        ext._validate_oracle_width(np.random.random((5, 9)), mlp)


def test_resolve_oracle_switches_by_matrix_width():
    rng = np.random.default_rng(1)
    n_base, n_extra = 6, 4
    X_base = rng.random((15, n_base))
    X_aug = np.hstack([X_base, rng.random((15, n_extra))])

    mlp_orig = MLPClassifier(max_iter=200, random_state=0)
    mlp_orig.fit(X_base, rng.integers(0, 2, size=15))
    mlp_onto = MLPClassifier(max_iter=200, random_state=1)
    mlp_onto.fit(X_aug, rng.integers(0, 2, size=15))

    ext = TrepanReloadedExtractor(ontology=object())
    ext._mlp_model_original = mlp_orig
    ext._mlp_model_onto = mlp_onto

    assert ext._resolve_oracle_for_matrix(X_base, mlp_onto) is mlp_orig
    assert ext._resolve_oracle_for_matrix(X_aug, mlp_orig) is mlp_onto
    preds = ext._mlp_predict(mlp_onto, X_base)
    assert len(preds) == len(X_base)


def test_expand_synthetic_requires_fitted_ontology_transformer():
    """Features OWL nunca podem ser inventadas a partir de uma matriz de referência."""
    rng = np.random.default_rng(30)
    n_base = 30
    base_names = [f"feat_{i}" for i in range(n_base)]
    onto_names = [f"onto_{n}_concept_score" for n in base_names[:6]]
    full_names = base_names + onto_names
    n_full = len(full_names)

    X_base = rng.random((20, n_base))
    X_ref = np.hstack([X_base, rng.random((20, len(onto_names)))])

    ext = TrepanReloadedExtractor(ontology=object())
    ext.ontology_active = True
    ext._matrix_feature_names = full_names
    ext._X_augmented_reference = X_ref
    ext.feature_centrality = {i: 0.4 for i in range(n_full)}

    with pytest.raises(ValueError, match="fitted|transform"):
        ext._expand_synthetic_by_schema(
            X_base, base_names, full_names, reference_X=X_ref
        )

    with pytest.raises(ValueError, match="fitted|transform|Transformador"):
        ext._expand_synthetic_with_semantic_features(
            rng.random((100, n_base)), base_names, full_names
        )


def test_is_pre_augmented_detects_gui_onto_metadata():
    ext = TrepanReloadedExtractor(ontology=object())
    base = [f"f{i}" for i in range(30)]
    aug = base + [f"onto_f0_concept_score", f"onto_f0_depth"]
    assert ext._is_pre_augmented_matrix(aug, base) is True
    assert ext._is_pre_augmented_matrix(base, base) is False


def test_class_names_for_tree_single_class_export():
    """export_text exige len(class_names) == len(tree.classes_)."""
    from sklearn.tree import DecisionTreeClassifier, export_text

    ext = TrepanReloadedExtractor(ontology=None)
    X = np.random.random((40, 5))
    y = np.zeros(40, dtype=int)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0)
    tree.fit(X, y)
    human_names = ["benign", "malignant"]
    export_names = ext._class_names_for_tree(tree, human_names)
    assert len(export_names) == 1
    assert export_names[0] == "benign"
    text = export_text(tree, feature_names=[f"f{i}" for i in range(5)], class_names=export_names)
    assert len(text) > 0


def test_ensure_training_label_diversity_from_mlp_original():
    rng = np.random.default_rng(0)
    n_base = 8
    X_base = rng.random((50, n_base))
    y_one = np.zeros(50, dtype=int)
    mlp_orig = MLPClassifier(max_iter=300, random_state=0)
    mlp_orig.fit(X_base, rng.integers(0, 2, size=50))

    ext = TrepanReloadedExtractor(ontology=object())
    ext._mlp_model_original = mlp_orig
    y_div = ext._ensure_training_label_diversity(
        y_one, X_base, X_base, rng.integers(0, 2, size=30),
        mlp_orig, [f"f{i}" for i in range(n_base)],
    )
    assert len(np.unique(y_div)) >= 2


def test_align_matrix_rejects_unfitted_arff_to_enriched_conversion():
    """A árvore enriquecida exige o mesmo transformador OWL ajustado no treino."""
    from sklearn.tree import DecisionTreeClassifier

    rng = np.random.default_rng(14)
    n_base, n_onto = 14, 30
    base_names = [f"attr_{i}" for i in range(n_base)]
    onto_names = [f"onto_attr_{i}_score" for i in range(n_onto)]
    full_names = base_names + onto_names

    X_base = rng.random((25, n_base))
    X_ref = np.hstack([X_base, rng.random((25, n_onto))])

    ext = TrepanReloadedExtractor(ontology=object())
    ext.ontology_active = True
    ext._base_feature_names = base_names
    ext._original_feature_names = base_names
    ext._matrix_feature_names = full_names
    ext._X_augmented_reference = X_ref

    with pytest.raises(ValueError, match="fitted|transform"):
        ext._expand_synthetic_by_schema(
            rng.random((80, n_base)), base_names, full_names, reference_X=X_ref
        )
