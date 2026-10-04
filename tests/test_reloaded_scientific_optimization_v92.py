from __future__ import annotations

import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.trepan_reloaded_historical import TrepanReloadedClassifier


def _oracle(X, y, seed=7):
    return LogisticRegression(max_iter=600, random_state=seed).fit(X, y)


def test_reload_generates_ontology_guided_mofn_candidates():
    rng = np.random.default_rng(41)
    X = rng.normal(size=(420, 5))
    y = ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0).astype(int) + (X[:, 2] > 0).astype(int) >= 2).astype(int)
    rel = np.eye(5)
    rel[:3, :3] = 1.0
    model = TrepanReloadedClassifier(
        max_nodes=7, max_depth=3, min_sample=260, max_queries=0,
        max_n=3, beam_width=1, max_features_per_node=3, random_state=41,
        semantic_candidate_budget=20,
    ).fit(
        X, y=y, feature_names=['a','b','c','d','e'],
        semantic_feature_weights=np.ones(5),
        semantic_feature_groups=['g','g','g',None,None],
        semantic_relatedness_matrix=rel,
    )
    assert model.semantic_candidate_generation_count_ > 0
    assert any(r.get('semantic_candidates_evaluated', 0) > 0 for r in model.semantic_split_audit_)
    assert model.semantic_audit_summary_['semantic_candidate_generation_count'] > 0


def test_reload_uses_active_semantic_boundary_queries_without_exceeding_budget():
    X, y = make_classification(
        n_samples=220, n_features=6, n_informative=5, n_redundant=0,
        class_sep=1.2, random_state=13,
    )
    oracle = _oracle(X, y, 13)
    rel = np.eye(6)
    rel[0,1] = rel[1,0] = 0.9
    model = TrepanReloadedClassifier(
        max_nodes=7, max_depth=3, min_sample=260, max_queries=700,
        max_n=3, random_state=13,
        semantic_active_query_fraction=0.70,
        semantic_active_pool_multiplier=4,
    ).fit(
        X, oracle=oracle, feature_names=[f'f{i}' for i in range(6)],
        semantic_feature_weights=np.array([1.4,1.3,1,1,1,1]),
        semantic_relatedness_matrix=rel,
    )
    assert model.membership_queries_ <= 700
    assert model.semantic_active_query_batches_ > 0
    assert model.semantic_active_query_selected_ > 0
    assert model.semantic_audit_summary_['active_query_batches'] > 0


def test_scientific_tuner_selects_common_capacity_and_semantic_params_using_training_only():
    from core.trepan_scientific_tuning import (
        ScientificTrepanSearchConfig,
        tune_scientific_trepan,
    )
    X, y = make_classification(
        n_samples=180, n_features=6, n_informative=5, n_redundant=0,
        class_sep=1.4, random_state=23,
    )
    oracle = _oracle(X, y, 23)
    rel = np.eye(6)
    rel[:3,:3] = 0.8
    np.fill_diagonal(rel, 1.0)
    base = ControlledTrepanConfig(
        max_nodes=9, max_depth=4, min_sample=130, max_queries=220,
        max_n=2, beam_width=2, max_features_per_node=6, random_state=23,
    )
    search = ScientificTrepanSearchConfig(
        cv_folds=2,
        cv_repeats=1,
        tune_structure=False,
        max_capacity_candidates=2,
        max_semantic_candidates=2,
        fidelity_target=0.85,
    )
    result = tune_scientific_trepan(
        X, y, oracle=oracle, feature_names=[f'f{i}' for i in range(6)],
        base_config=base,
        semantic_feature_weights=np.array([1.2,1.2,1.2,1,1,1]),
        semantic_feature_groups=['g','g','g',None,None,None],
        semantic_relatedness_matrix=rel,
        search=search,
    )
    assert result['test_used_for_selection'] is False
    assert result['selection_scope'] == 'training_cv_only'
    assert result['selected_config']['max_queries'] == result['common_capacity']['max_queries']
    assert result['selected_config']['max_nodes'] == result['common_capacity']['max_nodes']
    assert result['selected_config']['semantic_gain_strength'] >= 0.0
    assert result['capacity_history']
    assert result['semantic_history']


def test_typed_ontology_relatedness_weights_stronger_relations_more():
    from core.ontology_semantic_graph import OntologySemanticGraph
    g = OntologySemanticGraph()
    g.nodes.update({'a','b','c'})
    g.edges['a'].update({'b','c'}); g.edges['b'].add('a'); g.edges['c'].add('a')
    g.edge_types[('a','b')].add('equivalent'); g.edge_types[('b','a')].add('equivalent')
    g.edge_types[('a','c')].add('domain'); g.edge_types[('c','a')].add('domain')
    assert g.relatedness('a','b') > g.relatedness('a','c') > 0.0


def test_gui_scientific_flow_contains_train_only_common_capacity_tuning():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / 'gui' / 'biuri_app_complete.py').read_text(encoding='utf-8')
    assert 'tune_scientific_trepan(' in source
    assert "self._eval_split_original['X_train']" in source
    assert "self._eval_split_original['X_test']" not in source[source.find('self._trepan_scientific_tuning = None'):source.find('trepan_train_kwargs = dict', source.find('self._trepan_scientific_tuning = None'))]
    assert "'canonical_max_nodes': int(common['max_nodes'])" in source


def test_production_report_records_train_only_scientific_tuning(tmp_path):
    import pandas as pd
    from sklearn.datasets import make_classification
    from core.production_training import train_production_dataframe
    from core.trepan_scientific_tuning import ScientificTrepanSearchConfig
    X, y = make_classification(n_samples=80, n_features=5, n_informative=4, n_redundant=0, random_state=51)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
    df['target'] = np.where(y == 1, 'yes', 'no')
    report = train_production_dataframe(
        df, target='target', out_dir=tmp_path, seed=51,
        trepan_search=ScientificTrepanSearchConfig(cv_folds=2, cv_repeats=1, tune_structure=False, max_capacity_candidates=1, max_semantic_candidates=1),
    )
    tuning = report['evaluation']['trepan_scientific_tuning']
    assert tuning['selection_scope'] == 'training_cv_only'
    assert tuning['test_used_for_selection'] is False
    assert report['manifest']['scientific_tuning'] is True
