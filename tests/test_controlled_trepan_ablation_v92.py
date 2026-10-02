import inspect

import numpy as np
from sklearn.datasets import make_classification


def _oracle_fixture(seed=17):
    from core.mlp_factory import build_mlp_for_data
    X, y = make_classification(
        n_samples=260, n_features=5, n_informative=4, n_redundant=0,
        class_sep=1.5, random_state=seed,
    )
    model = build_mlp_for_data(X, y, random_state=seed).fit(X, y)
    return X, y, model


def test_controlled_pair_neutral_reloaded_is_exact_original():
    from core.controlled_trepan_experiment import (
        ControlledTrepanConfig,
        fit_controlled_trepan_pair,
    )
    X, y, oracle = _oracle_fixture()
    cfg = ControlledTrepanConfig(
        max_nodes=9, max_depth=4, min_sample=180, max_queries=1200,
        max_n=2, beam_width=2, random_state=17,
    )
    pair = fit_controlled_trepan_pair(
        X, y, oracle=oracle, feature_names=[f'f{i}' for i in range(X.shape[1])],
        config=cfg,
    )
    assert pair.audit['same_oracle'] is True
    assert pair.audit['same_seed'] is True
    assert pair.audit['same_tree_budget'] is True
    assert pair.audit['only_experimental_variable'] == 'semantic_ontology_extension'
    assert pair.reloaded.export_rules() == pair.original.export_rules()
    np.testing.assert_array_equal(pair.reloaded.predict(X), pair.original.predict(X))
    assert pair.original.max_queries == pair.reloaded.max_queries
    assert pair.original.min_sample == pair.reloaded.min_sample


def test_augmented_arm_uses_same_original_oracle_through_projection():
    from core.controlled_trepan_experiment import (
        ControlledTrepanConfig,
        fit_controlled_trepan_pair,
    )
    X, y, oracle = _oracle_fixture(21)
    onto = (X[:, 0] > np.median(X[:, 0])).astype(float).reshape(-1, 1)
    X_aug = np.column_stack([X, onto])
    names = [f'f{i}' for i in range(X.shape[1])] + ['onto_high_f0']

    def projector(rows):
        rows = np.asarray(rows, dtype=float).copy()
        rows[:, -1] = (rows[:, 0] > np.median(X[:, 0])).astype(float)
        return rows

    cfg = ControlledTrepanConfig(
        max_nodes=9, max_depth=4, min_sample=180, max_queries=1200,
        max_n=2, beam_width=2, random_state=21,
    )
    pair = fit_controlled_trepan_pair(
        X, y, oracle=oracle, feature_names=[f'f{i}' for i in range(X.shape[1])],
        reloaded_X_train=X_aug,
        reloaded_feature_names=names,
        original_feature_indices=list(range(X.shape[1])),
        semantic_feature_weights=np.r_[np.ones(X.shape[1]), 1.5],
        query_projector=projector,
        config=cfg,
    )
    assert pair.audit['same_oracle'] is True
    assert pair.audit['reloaded_oracle_adapter'] == 'OriginalOracleProjection'
    base_pred = oracle.predict(X)
    aug_pred = pair.reloaded_oracle_for_audit.predict(X_aug)
    np.testing.assert_array_equal(base_pred, aug_pred)
    assert pair.reloaded.algorithm_family_ == 'historical_trepan'


def test_final_evaluation_is_paired_and_locked_test_not_used_for_fit():
    from core.controlled_trepan_experiment import (
        ControlledTrepanConfig,
        evaluate_controlled_trepan_pair,
        fit_controlled_trepan_pair,
    )
    X, y, oracle = _oracle_fixture(23)
    train, test = np.arange(190), np.arange(190, len(X))
    cfg = ControlledTrepanConfig(
        max_nodes=7, max_depth=3, min_sample=150, max_queries=900,
        max_n=2, random_state=23,
    )
    pair = fit_controlled_trepan_pair(
        X[train], y[train], oracle=oracle,
        feature_names=[f'f{i}' for i in range(X.shape[1])], config=cfg,
    )
    report = evaluate_controlled_trepan_pair(pair, X[test], y[test])
    assert report['protocol_audit']['test_used_for_selection'] is False
    assert report['protocol_audit']['final_test_evaluated'] is True
    assert report['protocol_audit']['events'][-1]['operation'] == 'paired_original_vs_reloaded_final_evaluation'
    assert 'original' in report['models'] and 'reloaded' in report['models']
    assert report['comparison']['delta_balanced_accuracy'] == 0.0


def test_ablation_with_owl_no_longer_changes_oracle_family():
    import core.ablation_study as module
    source = inspect.getsource(module.run_dataset_ablation)
    assert 'ResidualOntologicalOracle' not in source
    assert 'build_oof_residual_feature_matrices' not in source
    assert 'TrepanReloadedClassifier' in source
    assert 'OriginalOracleProjection' in source


def test_paired_owl_analysis_aggregates_repeats_by_dataset():
    from core.ablation_study import paired_owl_analysis
    rows = []
    for dataset, diffs in {'a': [0.10, 0.30], 'b': [-0.10, -0.10]}.items():
        for repeat, delta in enumerate(diffs):
            base = 0.70
            rows.append({'dataset': dataset, 'repeat': repeat, 'model': 'TREPAN Reloaded — sem OWL',
                         'accuracy': base, 'balanced_accuracy': base, 'macro_f1': base})
            rows.append({'dataset': dataset, 'repeat': repeat, 'model': 'TREPAN Reloaded — com OWL',
                         'accuracy': base + delta, 'balanced_accuracy': base + delta, 'macro_f1': base + delta})
    result = paired_owl_analysis(rows, random_state=1)
    assert result['unit_of_analysis'] == 'dataset'
    assert result['n_datasets'] == 2
    # dataset a => +0.20; dataset b => -0.10; média por dataset => +0.05
    assert np.isclose(result['accuracy']['mean_difference'], 0.05)
    assert len(result['accuracy']['differences']) == 2


def test_oracle_health_gate_compares_dummy_and_c45_only_on_training_cv():
    from core.controlled_trepan_experiment import oracle_health_gate
    from core.mlp_factory import build_mlp_for_data
    X, y = make_classification(
        n_samples=240, n_features=8, n_informative=6, class_sep=1.8,
        random_state=31,
    )
    estimator = build_mlp_for_data(X, y, random_state=31)
    report = oracle_health_gate(
        estimator, X, y, random_state=31, cv_folds=3,
        dummy_margin=0.02, c45_margin=0.10,
    )
    assert report['evaluation_scope'] == 'training_cv_only'
    assert 'mlp_cv_balanced_accuracy' in report
    assert 'c45_cv_balanced_accuracy' in report
    assert 'dummy_cv_balanced_accuracy' in report
    assert report['valid'] is True


def test_paired_analysis_excludes_invalid_oracle_dataset():
    from core.ablation_study import paired_owl_analysis
    rows = []
    for dataset, valid, delta in [('good', True, 0.10), ('bad', False, 0.90)]:
        rows.extend([
            {'dataset': dataset, 'repeat': 0, 'model': 'TREPAN Reloaded — sem OWL',
             'accuracy': .70, 'balanced_accuracy': .70, 'macro_f1': .70,
             'oracle_valid': valid},
            {'dataset': dataset, 'repeat': 0, 'model': 'TREPAN Reloaded — com OWL',
             'accuracy': .70 + delta, 'balanced_accuracy': .70 + delta, 'macro_f1': .70 + delta,
             'oracle_valid': valid},
        ])
    result = paired_owl_analysis(rows, random_state=1)
    assert result['n_datasets'] == 1
    assert result['excluded_datasets'] == ['bad']
    assert np.isclose(result['accuracy']['mean_difference'], 0.10)
