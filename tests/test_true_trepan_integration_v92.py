from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from core.trepan_original import TrepanOriginalClassifier, TrepanOriginalExtractor
from core.trepan_reloaded_extractor import TrepanReloadedExtractor
from counterfactuals.global_rules import extract_global_rules, analyse_global_rules


ROOT = Path(__file__).resolve().parents[1]


def _oracle_data(seed=11):
    X, y = make_classification(
        n_samples=90, n_features=5, n_informative=4, n_redundant=0,
        class_sep=1.5, random_state=seed,
    )
    oracle = LogisticRegression(max_iter=1000, random_state=seed).fit(X, y)
    return X, y, oracle


def test_compat_extractor_returns_historical_trepan_not_cart():
    X, y, oracle = _oracle_data()
    extractor = TrepanOriginalExtractor(random_state=3)
    report = extractor.extract_tree(
        oracle, X, y, sample_size=300,
        feature_names=[f'x{i}' for i in range(X.shape[1])],
        class_names=['não', 'sim'],
        training_limits={'max_nodes': 9, 'max_depth': 4, 'min_sample': 140, 'max_queries': 500},
    )
    assert isinstance(extractor.explainer_tree, TrepanOriginalClassifier)
    assert not hasattr(extractor.explainer_tree, 'tree_')
    assert extractor.last_audit['historical_core'] is True
    assert extractor.last_audit['distilled_cart'] is False
    assert 'TREPAN Original histórico' in report


def test_reloaded_without_ontology_delegates_to_same_historical_core():
    X, y, oracle = _oracle_data(13)
    reloaded = TrepanReloadedExtractor(ontology=None)
    reloaded.extract_tree(
        oracle, X, y, sample_size=260,
        feature_names=[f'x{i}' for i in range(X.shape[1])],
        class_names=['0', '1'],
        X_train=X, y_train=y,
    )
    assert isinstance(reloaded.explainer_tree, TrepanOriginalClassifier)
    assert isinstance(reloaded._original_extractor, TrepanOriginalExtractor)
    assert reloaded.last_audit['historical_core'] is True


def test_counterfactual_global_rules_preserve_m_of_n_conditions():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(260, 3))

    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, values):
            a = np.asarray(values)
            votes = (a[:, 0] > 0).astype(int) + (a[:, 1] > 0).astype(int) + (a[:, 2] > 0).astype(int)
            return (votes >= 2).astype(int)

    oracle = Oracle()
    tree = TrepanOriginalClassifier(
        max_nodes=11, max_depth=4, min_sample=180, max_queries=2200,
        max_n=3, random_state=7,
    ).fit(X, oracle=oracle, feature_names=['a', 'b', 'c'])
    rules = extract_global_rules(tree, ['a', 'b', 'c'], {0: 'não', 1: 'sim'})
    assert rules
    assert any(condition.kind == 'm_of_n' for rule in rules for condition in rule.conditions)
    result = analyse_global_rules(
        tree, oracle, X, ['a', 'b', 'c'], class_labels={0: 'não', 1: 'sim'},
        remove_fragile=False,
    )
    assert result['aggregate_metrics']['global_fidelity'] >= 0.80
    assert result['aggregate_metrics']['n_rules'] == len(rules)


def test_primary_integration_paths_do_not_import_legacy_trepan_extractor():
    paths = [
        ROOT / 'gui' / 'biuri_app_complete.py',
        ROOT / 'counterfactuals' / 'service.py',
        ROOT / 'counterfactuals' / 'pipelines' / 'pipeline_train.py',
        ROOT / 'counterfactuals' / 'pipelines' / 'pipeline_improve.py',
        ROOT / 'core' / 'trepan_reloaded_extractor.py',
    ]
    for path in paths:
        source = path.read_text(encoding='utf-8')
        assert 'from core.trepan_extractor import TREPANExtractor' not in source, path
        assert 'TREPANExtractor()' not in source, path


def test_reloaded_source_deploys_historical_core_not_canonical_or_redistilled_cart():
    source = (ROOT / 'core' / 'trepan_reloaded_extractor.py').read_text(encoding='utf-8')
    assert 'TrepanReloadedClassifier(' in source
    assert 'historical_trepan_reloaded' in source
    assert 'CanonicalTrepanClassifier(' not in source


def test_gui_tree_widget_declares_native_historical_support():
    source = (ROOT / 'gui' / 'pyqt_tree_widget.py').read_text(encoding='utf-8')
    assert '_build_historical_trepan_structure' in source
    assert "hasattr(self.tree_model, 'root_')" in source


def test_inference_explain_returns_native_m_of_n_path_not_all_rules():
    import pandas as pd
    from core.inference import Predictor

    rng = np.random.default_rng(19)
    X = rng.normal(size=(220, 3))

    class Oracle:
        classes_ = np.array([0, 1])
        def predict(self, values):
            a = np.asarray(values)
            votes = ((a[:, 0] > 0).astype(int) + (a[:, 1] > 0).astype(int) + (a[:, 2] > 0).astype(int))
            return (votes >= 2).astype(int)
        def predict_proba(self, values):
            pred = self.predict(values)
            return np.column_stack([1 - pred, pred]).astype(float)

    oracle = Oracle()
    tree = TrepanOriginalClassifier(
        max_nodes=9, max_depth=4, min_sample=160, max_queries=1800,
        max_n=3, random_state=19,
    ).fit(X, oracle=oracle, feature_names=['a', 'b', 'c'])
    predictor = Predictor(oracle, None, [0, 1], {}, tree)
    result = predictor.explain(pd.DataFrame([X[0]], columns=['a', 'b', 'c']))

    assert result['available'] is True
    assert result['tree_family'] == 'trepan_historical'
    assert 'rules' not in result
    assert 'path' in result and isinstance(result['path'], list)
    assert 'leaf' in result
    for step in result['path']:
        assert step['branch'] in {'sim', 'não'}
        assert step['m'] >= 1
        assert step['n'] >= step['m']
