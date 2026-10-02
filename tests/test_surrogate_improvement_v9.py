"""Regressões científicas da melhoria contrafactual V9."""
import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from counterfactuals.surrogate_improvement import (
    CandidateParameters,
    ImprovementSearchConfig,
    prepare_counterfactual_augmentation,
    select_and_refit_improved_surrogate,
)


class ThresholdModel:
    classes_ = np.asarray([0, 1])

    def __init__(self, threshold, nodes=3):
        self.threshold = float(threshold)
        self.tree_ = SimpleNamespace(node_count=int(nodes))

    def predict(self, values):
        values = np.asarray(values, dtype=float)
        return (values[:, 0] > self.threshold).astype(int)

    def predict_proba(self, values):
        pred = self.predict(values)
        confidence = np.full(len(pred), 0.9)
        return np.column_stack([
            np.where(pred == 0, confidence, 1 - confidence),
            np.where(pred == 1, confidence, 1 - confidence),
        ])

    def get_depth(self):
        return 1

    def get_n_leaves(self):
        return 2


def test_density_is_a_real_gate_and_cf_weight_is_bounded():
    X = np.asarray([[-0.20], [-0.10], [-0.01], [0.05], [0.10], [0.20]])
    oracle = ThresholdModel(0.0)
    tree = ThresholdModel(0.5)
    y = oracle.predict(X)
    cfs = [
        {'cf': np.asarray([0.01]), 'original_class': 0},
        {'cf': np.asarray([50.0]), 'original_class': 0},
    ]
    config = ImprovementSearchConfig(
        strict_density_gate=True,
        sample_size=100,
        max_cf_ratio=1.0,
        anchor_ratio=0.2,
        max_cfs_grid=(2,),
        cf_types_grid=('A+B+C',),
        cf_weight_ratio_grid=(0.05,),
    )
    batch = prepare_counterfactual_augmentation(
        oracle,
        tree,
        X,
        y,
        cfs,
        parameters=CandidateParameters(2, 'A+B+C', 0.05),
        config=config,
    )
    assert batch.stats['density_rejected'] == 1
    assert batch.stats['used'] == 1
    assert batch.stats['total_cf_weight'] <= batch.stats['cf_weight_cap'] + 1e-12


def test_selector_attributes_gain_to_cf_and_never_uses_locked_test(monkeypatch):
    X = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    oracle = ThresholdModel(0.0)
    y = oracle.predict(X)
    current = ThresholdModel(1.0, nodes=3)

    def fake_pool(*args, **kwargs):
        return ([
            {'cf': np.asarray([0.1]), 'original_class': 0, 'data_role': 'development_fit_only'},
            {'cf': np.asarray([-0.1]), 'original_class': 1, 'data_role': 'development_fit_only'},
        ], {'data_role': 'development_fit_only', 'test_queried': False})

    monkeypatch.setattr(
        'counterfactuals.surrogate_improvement.generate_counterfactual_pool',
        fake_pool,
    )

    def builder(request):
        if request.extra_X is None:
            return ThresholdModel(0.65, nodes=3), {'control': True}
        return ThresholdModel(0.0, nodes=3), {'used_cf': True}

    config = ImprovementSearchConfig(
        strict_density_gate=False,
        max_cf_ratio=1.0,
        sample_size=100,
        max_cfs_grid=(2,),
        cf_types_grid=('A+B+C',),
        cf_weight_ratio_grid=(0.05,),
        max_complexity_ratio=2.0,
    )
    result = select_and_refit_improved_surrogate(
        model_name='TREPAN Original',
        oracle=oracle,
        original_tree=current,
        X_development=X,
        y_development=y,
        feature_names=['x'],
        candidate_builder=builder,
        config=config,
    )
    assert result['accepted'] is True
    assert result['deployed_tree'].threshold == 0.0
    assert result['acceptance_gate']['checks']['counterfactual_contribution'] is True
    assert result['acceptance_gate']['decision_scope'] == 'independent_internal_acceptance_holdout'
    assert result['partition_sizes']['configuration_selection'] > 0
    assert result['partition_sizes']['acceptance_holdout'] > 0
    assert result['protocol_audit']['test_used_for_selection'] is False
    assert result['locked_test_used'] is False
    assert result['final_test_evaluated'] is False


def test_acceptance_holdout_can_reject_a_configuration_that_won_selection(monkeypatch):
    X = np.linspace(-2.0, 2.0, 80).reshape(-1, 1)
    oracle = ThresholdModel(0.0)
    y = oracle.predict(X)
    current = ThresholdModel(0.8)

    monkeypatch.setattr(
        'counterfactuals.surrogate_improvement.generate_counterfactual_pool',
        lambda *args, **kwargs: ([
            {'cf': np.asarray([0.1]), 'original_class': 0},
            {'cf': np.asarray([-0.1]), 'original_class': 1},
        ], {'data_role': 'development_fit_only', 'test_queried': False}),
    )

    def builder(request):
        if request.extra_X is None:
            return ThresholdModel(0.65), {}
        # A configuração parece perfeita na seleção (48 linhas de ajuste),
        # mas falha quando é refeita nos 64 casos que antecedem a auditoria.
        threshold = 0.0 if len(request.X_train) < 60 else 1.5
        return ThresholdModel(threshold), {}

    config = ImprovementSearchConfig(
        strict_density_gate=False,
        max_cf_ratio=1.0,
        sample_size=100,
        max_cfs_grid=(2,),
        cf_types_grid=('A+B+C',),
        cf_weight_ratio_grid=(0.05,),
        max_complexity_ratio=2.0,
    )
    result = select_and_refit_improved_surrogate(
        model_name='TREPAN Original',
        oracle=oracle,
        original_tree=current,
        X_development=X,
        y_development=y,
        feature_names=['x'],
        candidate_builder=builder,
        config=config,
    )
    assert result['selection_candidate_metrics']['balanced_accuracy'] == pytest.approx(1.0)
    assert result['accepted'] is False
    assert result['status'] == 'rejected_by_quality_gate'
    assert result['deployed_tree'] is current
    assert result['locked_test_used'] is False


def _fake_result(name, tree):
    metrics = {
        'accuracy': 0.7,
        'balanced_accuracy': 0.7,
        'macro_f1': 0.7,
        'minority_recall': 0.7,
        'fidelity': 0.8,
        'depth': 1,
        'leaves': 2,
        'nodes': 3,
    }
    return {
        'model_name': name,
        'status': 'rejected_by_quality_gate',
        'accepted': False,
        'original_tree': tree,
        'candidate_tree': tree,
        'deployed_tree': tree,
        'baseline_metrics': metrics,
        'matched_no_cf_metrics': metrics,
        'candidate_metrics': metrics,
        'selected_augmentation': {},
    }


def test_service_uses_distinct_feature_spaces(monkeypatch):
    from counterfactuals.service import improve_surrogates_from_session

    calls = []

    def fake_selector(**kwargs):
        calls.append((kwargs['model_name'], kwargs['X_development'].shape[1], kwargs['oracle']))
        return _fake_result(kwargs['model_name'], kwargs['original_tree'])

    monkeypatch.setattr(
        'counterfactuals.surrogate_improvement.select_and_refit_improved_surrogate',
        fake_selector,
    )
    original_oracle = ThresholdModel(0.0)
    reloaded_oracle = ThresholdModel(0.0)
    original_tree = ThresholdModel(0.5)
    reloaded_tree = ThresholdModel(0.2)
    session = {
        'class_names': ['0', '1'],
        'class_labels': {0: '0', 1: '1'},
        'improvement_contexts': {
            'trepan_original': {
                'display_name': 'TREPAN Original', 'oracle': original_oracle,
                'tree': original_tree, 'X_development': np.ones((20, 2)),
                'y_development': np.asarray([0, 1] * 10),
                'feature_names': ['a', 'b'], 'ontology_active': False,
                'mirrors_original': False,
            },
            'trepan_reloaded': {
                'display_name': 'TREPAN Reloaded', 'oracle': reloaded_oracle,
                'tree': reloaded_tree, 'X_development': np.ones((20, 3)),
                'y_development': np.asarray([0, 1] * 10),
                'feature_names': ['a', 'b', 'onto_c'], 'ontology_active': False,
                'mirrors_original': False,
            },
        },
    }
    output = improve_surrogates_from_session(
        session, options={'target_model': 'both', 'max_cfs_grid': (2,)}
    )
    assert [(name, width) for name, width, _ in calls] == [
        ('TREPAN Original', 2), ('TREPAN Reloaded', 3)
    ]
    assert calls[0][2] is original_oracle
    assert calls[1][2] is reloaded_oracle
    assert output['test_used_for_selection'] is False
    assert output['prior_interactive_cf_reused'] is False


def test_service_mirrors_reloaded_without_accepted_ontology(monkeypatch):
    from counterfactuals.service import improve_surrogates_from_session

    calls = []

    def fake_selector(**kwargs):
        calls.append(kwargs['model_name'])
        return _fake_result(kwargs['model_name'], kwargs['original_tree'])

    monkeypatch.setattr(
        'counterfactuals.surrogate_improvement.select_and_refit_improved_surrogate',
        fake_selector,
    )
    oracle = ThresholdModel(0.0)
    tree = ThresholdModel(0.2)
    base = {
        'oracle': oracle, 'tree': tree,
        'X_development': np.ones((20, 1)),
        'y_development': np.asarray([0, 1] * 10),
        'feature_names': ['a'], 'ontology_active': False,
    }
    session = {
        'class_names': ['0', '1'],
        'improvement_contexts': {
            'trepan_original': dict(base, display_name='TREPAN Original', mirrors_original=False),
            'trepan_reloaded': dict(base, display_name='TREPAN Reloaded', mirrors_original=True),
        },
    }
    output = improve_surrogates_from_session(session, options={'target_model': 'both'})
    assert calls == ['TREPAN Original']
    assert output['results']['trepan_reloaded']['status'] == 'mirrored_no_ontology'
    assert output['results']['trepan_reloaded']['deployed_tree'] is tree


def test_c45_cannot_be_relabelled_as_counterfactual_surrogate():
    from counterfactuals.service import improve_surrogates_from_session

    with pytest.raises(ValueError, match='baseline supervisionado'):
        improve_surrogates_from_session(
            {'improvement_contexts': {'placeholder': {}}},
            options={'target_model': 'c45'},
        )


def test_zero_internal_cf_budget_really_disables_the_reloaded_batch():
    from core.plausible_counterfactuals import (
        PlausibleCFConfig,
        generate_plausible_boundary_counterfactuals,
    )

    X = np.asarray([[-1.0], [-0.5], [0.5], [1.0]])
    vectors, weights, audit = generate_plausible_boundary_counterfactuals(
        ThresholdModel(0.0),
        X,
        config=PlausibleCFConfig(max_counterfactuals=0),
    )
    assert vectors.shape == (0, 1)
    assert weights.shape == (0,)
    assert audit['reason'] == 'disabled_for_matched_control'
    assert audit['test_used'] is False


def test_gui_wires_the_improvement_dialog_without_reusing_generic_cfs():
    root = Path(__file__).resolve().parents[1]
    app_source = (root / 'gui' / 'biuri_app_complete.py').read_text(encoding='utf-8')
    worker_source = (root / 'gui' / 'counterfactual_worker.py').read_text(encoding='utf-8')
    dialog_source = (root / 'gui' / 'surrogate_improvement_dialog.py').read_text(
        encoding='utf-8'
    )
    ast.parse(app_source)
    ast.parse(worker_source)
    ast.parse(dialog_source)
    assert 'SurrogateImprovementDialog' in app_source
    assert "options=dialog.options()" in app_source
    assert "'improvement_contexts': improvement_contexts" in app_source
    assert "reloaded_development_context.pop('X_test', None)" in app_source
    assert 'Trepan-Original — melhorada por CF' in app_source
    assert 'Trepan-Reloaded — melhorada por CF' in app_source
    assert 'options=self.options' in worker_source
    assert 'C4.5 não aparece porque não é um substituto do MLP' in dialog_source
