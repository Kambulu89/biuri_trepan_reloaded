import numpy as np
import pytest

from core.trepan_original import (
    ConstraintSet,
    FeatureDistributionModel,
    Literal,
    MofNTest,
    TrepanOriginalClassifier,
)


class TwoOfThreeOracle:
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return ((X[:, :3] > 0).sum(axis=1) >= 2).astype(int)


def test_draw_instance_respects_path_constraints():
    rng = np.random.default_rng(11)
    X = rng.normal(size=(400, 4))
    model = FeatureDistributionModel(random_state=11).fit(X)
    constraints = ConstraintSet()
    constraints.add(MofNTest(1, (Literal(0, 0.0, True),)), True)
    constraints.add(MofNTest(2, (Literal(1, 0.0, True), Literal(2, 0.0, True), Literal(3, 0.0, True))), True)
    Q = model.draw(250, constraints)
    assert np.all(Q[:, 0] > 0)
    assert np.all((Q[:, 1:] > 0).sum(axis=1) >= 2)


def test_min_sample_is_enforced_per_expanded_node():
    rng = np.random.default_rng(2)
    X = rng.normal(size=(80, 4))
    oracle = TwoOfThreeOracle()
    t = TrepanOriginalClassifier(
        max_nodes=7, max_depth=3, min_sample=120, max_queries=1000,
        max_n=3, random_state=2,
    ).fit(X, oracle=oracle)
    expanded = [a for a in t.node_audit_ if a.get('expanded')]
    assert expanded
    assert all(a['decision_sample_size'] >= 120 for a in expanded)
    assert t.membership_queries_ > 0


def test_best_first_priority_formula_is_audited():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(220, 5))
    oracle = TwoOfThreeOracle()
    t = TrepanOriginalClassifier(max_nodes=11, min_sample=160, max_queries=2500, random_state=3).fit(X, oracle=oracle)
    for item in t.node_audit_:
        if 'priority' in item:
            assert item['priority'] == pytest.approx(item['reach'] * (1.0 - item['fidelity']))


def test_known_m_of_n_rule_is_recovered_with_high_fidelity():
    rng = np.random.default_rng(4)
    X = rng.normal(size=(500, 5))
    oracle = TwoOfThreeOracle()
    y = oracle.predict(X)
    t = TrepanOriginalClassifier(
        max_nodes=15, max_depth=5, min_sample=350, max_queries=5000,
        max_n=3, beam_width=2, random_state=4,
    ).fit(X, oracle=oracle, feature_names=[f'x{i}' for i in range(5)])
    assert t.node_count_ <= 15
    assert np.mean(t.predict(X) == y) >= 0.90
    assert any(a.get('n', 1) >= 2 for a in t.split_audit_)


def test_same_seed_produces_same_tree_and_query_count():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(250, 4))
    oracle = TwoOfThreeOracle()
    kwargs = dict(max_nodes=9, min_sample=180, max_queries=1800, random_state=8)
    a = TrepanOriginalClassifier(**kwargs).fit(X, oracle=oracle)
    b = TrepanOriginalClassifier(**kwargs).fit(X, oracle=oracle)
    assert a.export_rules() == b.export_rules()
    assert a.membership_queries_ == b.membership_queries_
