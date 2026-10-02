from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from core.metrics_comparator import MetricsComparator
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.trepan_reloaded_extractor import TrepanReloadedExtractor


class _SameSurrogate:
    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.predict(X)


def test_same_predictions_use_exact_same_control_fidelity_for_original_and_reloaded():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(90, 3))
    y = (X[:, 0] - 0.4 * X[:, 1] > 0).astype(int)
    mlp = LogisticRegression(random_state=4).fit(X[:60], y[:60])
    tree = _SameSurrogate(mlp)

    comp = MetricsComparator(n_bootstrap=25, random_state=4)
    fidelity = comp._calculate_fidelity_metrics(
        mlp, tree, tree, None, X[60:], y[60:],
        feature_names=['a', 'b', 'c'],
        X_test_reloaded=X[60:],
        mlp_model_reloaded=None,
        feature_names_reloaded=['a', 'b', 'c'],
    )

    assert fidelity['trepan_original']['overall_fidelity'] == 1.0
    assert fidelity['trepan_reloaded']['overall_fidelity'] == 1.0
    assert fidelity['trepan_original']['fidelity_to_mlp_original'] == 1.0
    assert fidelity['trepan_reloaded']['fidelity_to_mlp_original'] == 1.0
    assert fidelity['trepan_original']['overall_fidelity_ci']['point_estimate'] == 1.0
    assert fidelity['trepan_reloaded']['overall_fidelity_ci']['point_estimate'] == 1.0


def test_rejected_or_fallback_teacher_is_not_exposed_as_mlp_ontological():
    base = object()
    assert MetricsComparator._should_expose_ontological_mlp('mlp_original', base, base) is False
    assert MetricsComparator._should_expose_ontological_mlp('mlp_original', base, object()) is False
    assert MetricsComparator._should_expose_ontological_mlp('mlp_onto', base, base) is False
    assert MetricsComparator._should_expose_ontological_mlp('mlp_onto', base, object()) is True


def test_semantic_split_audit_reports_usage_and_decision_impact_without_cart():
    rng = np.random.default_rng(12)
    X = rng.normal(size=(650, 3))
    y = (1.10 * X[:, 0] + 0.92 * X[:, 1] + rng.normal(0, 0.5, len(X)) > 0).astype(int)

    model = TrepanReloadedClassifier(
        max_nodes=7, max_depth=3, min_sample=320, max_queries=0,
        max_n=1, random_state=12,
    ).fit(
        X, y=y, feature_names=['f0', 'semantic_f1', 'f2'],
        semantic_feature_weights=np.array([1.0, 3.8, 1.0]),
    )

    summary = model.semantic_audit_summary_
    assert summary['evaluated_splits'] >= 1
    assert 0.0 <= summary['ontology_usage_rate'] <= 1.0
    assert 0.0 <= summary['semantic_decision_impact'] <= 1.0
    assert model.semantic_split_audit_
    for row in model.semantic_split_audit_:
        assert 'data_only_test' in row
        assert 'semantic_test' in row
        assert 'semantic_bonus' in row
        assert 'decision_changed' in row
        assert 'ontology_influenced' in row
    assert not hasattr(model, 'tree_')


def test_property_collection_works_when_ontology_exposes_only_data_properties():
    class Entity:
        def __init__(self, name):
            self.name = name
            self.iri = f'urn:test:{name}'

    class Ontology:
        def data_properties(self):
            return [Entity('p1'), Entity('p2')]

        def object_properties(self):
            return []

    extractor = TrepanReloadedExtractor(ontology=Ontology())
    props = extractor._collect_ontology_properties()
    assert [p.name for p in props] == ['p1', 'p2']

def test_equal_feature_weights_can_still_use_ontology_group_coherence():
    rng = np.random.default_rng(31)
    X = rng.normal(size=(500, 4))
    y = ((X[:, :3] > 0).sum(axis=1) >= 2).astype(int)

    model = TrepanReloadedClassifier(
        max_nodes=7, max_depth=3, min_sample=260, max_queries=0,
        max_n=3, random_state=31, semantic_group_strength=0.25,
    ).fit(
        X,
        y=y,
        feature_names=['a', 'b', 'c', 'd'],
        semantic_feature_weights=np.ones(4),
        semantic_feature_groups=['morphology', 'morphology', 'morphology', 'other'],
    )

    assert model.reload_extension_active_ is True
    assert model.semantic_split_audit_
    # Mesmo com pesos individuais todos iguais, uma regra m-of-n coerente pode
    # ser semanticamente reforçada pela ontologia.
    assert any(row.get('semantic_group_factor', 1.0) > 1.0 for row in model.semantic_split_audit_)
    assert model.semantic_audit_summary_['ontology_usage_rate'] > 0.0

def test_projected_original_oracle_is_not_mislabelled_as_ontological_teacher():
    class ProjectedOriginal:
        ORACLE_TYPE = 'projected_original'
        n_features_in_ = 5

    assert MetricsComparator._reloaded_fidelity_reference(
        ProjectedOriginal(), [f'f{i}' for i in range(5)], n_matrix_features=5
    ) == 'mlp_original'
