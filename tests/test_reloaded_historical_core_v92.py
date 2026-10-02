import inspect

import numpy as np

from core.trepan_original import TrepanOriginalClassifier


def _two_of_three_oracle():
    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, X):
            X = np.asarray(X, dtype=float)
            return ((X[:, :3] > 0).sum(axis=1) >= 2).astype(int)
    return Oracle()


def test_reloaded_historical_neutral_semantics_matches_original_exactly():
    from core.trepan_reloaded_historical import TrepanReloadedClassifier

    rng = np.random.default_rng(91)
    X = rng.normal(size=(260, 4))
    oracle = _two_of_three_oracle()
    kwargs = dict(
        max_nodes=11,
        max_depth=4,
        min_sample=180,
        max_queries=2200,
        max_n=3,
        beam_width=2,
        random_state=91,
    )
    original = TrepanOriginalClassifier(**kwargs).fit(
        X, oracle=oracle, feature_names=['a', 'b', 'c', 'd']
    )
    reloaded = TrepanReloadedClassifier(**kwargs).fit(
        X,
        oracle=oracle,
        feature_names=['a', 'b', 'c', 'd'],
        semantic_feature_weights=np.ones(X.shape[1]),
    )

    assert reloaded.export_rules() == original.export_rules()
    np.testing.assert_array_equal(reloaded.predict(X), original.predict(X))
    assert reloaded.membership_queries_ == original.membership_queries_
    assert reloaded.algorithm_family_ == 'historical_trepan'
    assert reloaded.reload_extension_active_ is False


def test_reloaded_semantic_weight_biases_split_search_without_cart():
    from core.trepan_reloaded_historical import TrepanReloadedClassifier

    rng = np.random.default_rng(12)
    X = rng.normal(size=(600, 3))
    # f0 tem sinal ligeiramente mais forte; f1 continua útil.
    margin = 1.15 * X[:, 0] + 0.85 * X[:, 1] + rng.normal(0, 0.55, len(X))
    y = (margin > 0).astype(int)

    neutral = TrepanReloadedClassifier(
        max_nodes=5, max_depth=2, min_sample=300, max_queries=0,
        max_n=1, random_state=12,
    ).fit(
        X, y=y, feature_names=['f0', 'onto_f1', 'f2'],
        semantic_feature_weights=np.ones(3),
    )
    semantic = TrepanReloadedClassifier(
        max_nodes=5, max_depth=2, min_sample=300, max_queries=0,
        max_n=1, random_state=12,
    ).fit(
        X, y=y, feature_names=['f0', 'onto_f1', 'f2'],
        semantic_feature_weights=np.array([1.0, 3.5, 1.0]),
    )

    assert not hasattr(semantic, 'tree_')
    assert semantic.reload_extension_active_ is True
    assert semantic.semantic_split_audit_
    assert any(item['semantic_weight'] > 1.0 for item in semantic.semantic_split_audit_)
    # O bias deve conseguir promover a feature ontológica para pelo menos um split.
    used = {
        lit.feature
        for node, test in semantic.iter_splits()
        for lit in test.literals
    }
    assert 1 in used
    assert semantic.algorithm_family_ == 'historical_trepan'


def test_reloaded_membership_queries_use_semantic_projector():
    from core.trepan_reloaded_historical import TrepanReloadedClassifier

    rng = np.random.default_rng(77)
    base = rng.normal(size=(180, 2))
    X = np.column_stack([base, base[:, 0] + base[:, 1]])
    calls = []

    def projector(rows):
        rows = np.asarray(rows, dtype=float).copy()
        rows[:, 2] = rows[:, 0] + rows[:, 1]
        calls.append(len(rows))
        return rows

    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, rows):
            rows = np.asarray(rows, dtype=float)
            # Se a projecção não for respeitada, o teste detecta inconsistência.
            assert np.allclose(rows[:, 2], rows[:, 0] + rows[:, 1], atol=1e-10)
            return (rows[:, 2] > 0).astype(int)

    model = TrepanReloadedClassifier(
        max_nodes=7, max_depth=3, min_sample=260, max_queries=1200,
        max_n=2, random_state=77,
    ).fit(
        X,
        oracle=Oracle(),
        feature_names=['a', 'b', 'onto_sum'],
        semantic_feature_weights=np.array([1.0, 1.0, 2.0]),
        query_projector=projector,
    )

    assert calls
    assert model.semantic_projected_query_count_ > 0
    assert model.membership_queries_ > 0
    assert model.reload_extension_active_ is True


def test_reloaded_main_owl_path_has_no_cart_or_soft_tree_refinement_calls():
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor

    source = inspect.getsource(TrepanReloadedExtractor.extract_tree_with_ontology)
    forbidden = [
        '_extract_faithful_reloaded_tree(',
        'refine_with_active_queries(',
        'SoftDecisionTreeClassifier(',
        'DecisionTreeClassifier(',
        '_fit_tree_with_semantic_tuning(',
        '_calibrate_onto_feature_bias_weight_internal(',
        'select_global_surrogate(',
        'redistill_to_crisp_tree(',
    ]
    for token in forbidden:
        assert token not in source, token

    assert '_extract_historical_reloaded_core(' in source


def test_primary_historical_reloaded_core_returns_native_model_and_is_pickle_safe():
    import pickle
    from core.trepan_reloaded_historical import TrepanReloadedClassifier
    from core.trepan_reloaded_extractor import TrepanReloadedExtractor

    rng = np.random.default_rng(123)
    base = rng.normal(size=(120, 2))
    X = np.column_stack([base, base[:, 0] + base[:, 1]])
    y = (X[:, 2] > 0).astype(int)

    class Oracle:
        n_features_in_ = 3
        classes_ = np.array([0, 1])
        def predict(self, rows):
            rows = np.asarray(rows, dtype=float)
            return (rows[:, 2] > 0).astype(int)
        def predict_proba(self, rows):
            p = self.predict(rows)
            return np.column_stack([1 - p, p]).astype(float)

    class Processor:
        is_fitted_ = True
        def transform_matrix(self, base_rows, base_names):
            base_rows = np.asarray(base_rows, dtype=float)
            return (
                np.column_stack([base_rows, base_rows[:, 0] + base_rows[:, 1]]),
                ['a', 'b', 'onto_sum'],
            )
        def prune_semantically_incoherent_rules(self, rules, *args, **kwargs):
            return rules, []

    extractor = TrepanReloadedExtractor(ontology=object())
    extractor.ontology_processor = Processor()
    extractor.feature_semantics = {
        0: {'semantic_group': 'base', 'mapping_score': 0.0},
        1: {'semantic_group': 'base', 'mapping_score': 0.0},
        2: {
            'semantic_group': 'ontology', 'mapping_score': 1.0,
            'ontology_derived': True, 'split_priority': 1.2,
        },
    }
    extractor._matrix_feature_names = ['a', 'b', 'onto_sum']
    extractor._base_feature_names = ['a', 'b']
    extractor._original_feature_names = ['a', 'b']
    extractor._X_augmented_reference = X
    extractor._training_limits = {
        'canonical_max_nodes': 7,
        'canonical_max_depth': 3,
        'historical_min_sample': 150,
        'historical_max_queries': 600,
        'canonical_m_of_n_max_n': 2,
    }
    # Este teste isola o núcleo; coerência OWL DL pertence ao quality gate próprio.
    extractor._validate_semantic_coherence = lambda *a, **k: {
        'issues': [], 'warnings': [], 'total_nodes': 0, 'validated_nodes': 0,
    }

    report = extractor._extract_historical_reloaded_core(
        labeling_oracle=Oracle(),
        mlp_model=Oracle(),
        X_train_augmented=X[:90],
        y_train_real=y[:90],
        X_eval=X[90:],
        y_eval_real=y[90:],
        tree_feature_names=['a', 'b', 'onto_sum'],
        base_feature_names=['a', 'b'],
        class_names=['não', 'sim'],
        sample_size=400,
    )

    assert isinstance(extractor.explainer_tree, TrepanReloadedClassifier)
    assert not hasattr(extractor.explainer_tree, 'tree_')
    assert extractor.last_audit['final_tree_family'] == 'historical_trepan_reloaded'
    assert extractor.last_audit['cart_used_for_final'] is False
    assert extractor.last_audit['soft_tree_used_for_final'] is False
    assert extractor.last_audit['external_test_used_for_selection'] is False
    assert extractor._last_active_query_info['implementation'] == 'historical_membership_queries_per_node'
    assert 'TREPAN-RELOADED' in report
    # O projector é de treino e não fica serializado como closure.
    pickle.dumps(extractor.explainer_tree)
