"""O TREPAN regista por que cada nó e a árvore pararam (observabilidade; não altera decisões)."""
import numpy as np
import pytest

from core.trepan_original import TrepanOriginalClassifier
from core.trepan_reloaded_historical import TrepanReloadedClassifier


class Oracle:
    classes_ = np.array([0, 1])

    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        return self.fn(np.asarray(X, float)).astype(int)


TWO_OF_THREE = Oracle(lambda X: (X[:, :3] > 0).sum(axis=1) >= 2)
CONSTANT = Oracle(lambda X: np.zeros(len(X)))


def _fit(cls=TrepanOriginalClassifier, oracle=TWO_OF_THREE, **over):
    X = np.random.default_rng(1).normal(size=(260, 5))
    kw = dict(max_nodes=11, max_depth=4, min_sample=180, max_queries=2200, max_n=3, beam_width=2, random_state=1)
    kw.update(over)
    return cls(**kw).fit(X, oracle=oracle, feature_names=list("abcde"))


def test_summary_has_all_fields_and_is_consistent():
    m = _fit()
    s = m.stop_summary_
    assert {"loop_end_reason", "node_budget", "nodes_before_pruning", "nodes_after_pruning", "unexpanded_nodes_waiting",
            "query_budget", "queries_used", "query_budget_exhausted", "max_depth", "stop_reasons"} <= set(s)
    assert s["nodes_after_pruning"] == m.node_count_ <= s["nodes_before_pruning"] <= s["node_budget"]
    assert s["queries_used"] == m.membership_queries_ and s["query_budget"] == 2200
    assert s["stop_reasons"] == m.stop_reasons_


def test_node_budget_exhaustion_is_reported():
    m = _fit(max_nodes=3)
    assert m.stop_summary_["loop_end_reason"] == "node_budget_exhausted"
    assert m.stop_summary_["unexpanded_nodes_waiting"] >= 1 and m.node_count_ <= 3


def test_pure_oracle_stops_because_nodes_are_pure():
    m = _fit(oracle=CONSTANT)
    assert m.node_count_ == 1
    assert m.stop_summary_["loop_end_reason"] == "no_expandable_nodes_left"
    assert m.stop_reasons_.get("pure_node", 0) >= 1
    assert any(e.get("stop_reason") == "pure_node" for e in m.node_audit_)


def test_max_depth_reason_is_recorded():
    m = _fit(max_depth=1, max_nodes=31)
    assert m.stop_reasons_.get("max_depth", 0) >= 1
    assert any(e.get("stop_reason") == "max_depth" for e in m.node_audit_)
    assert m.stop_summary_["max_depth"] == 1


def test_query_budget_reason_when_budget_is_too_small():
    m = _fit(min_sample=1000, max_queries=10)
    assert m.stop_summary_["query_budget_exhausted"] is True or m.stop_reasons_.get("query_budget_before_min_sample", 0) >= 1


def test_every_unexpanded_decided_node_has_a_reason():
    m = _fit(max_nodes=31)
    decided = [e for e in m.node_audit_ if "decision_sample_size" in e]
    assert decided
    for e in decided:
        assert e["expanded"] or e.get("stop_reason"), e


def test_instrumentation_is_deterministic_and_does_not_change_the_tree():
    a, b = _fit(), _fit()
    assert a.export_rules() == b.export_rules() and a.stop_summary_ == b.stop_summary_
    X = np.random.default_rng(9).normal(size=(80, 5))
    np.testing.assert_array_equal(a.predict(X), b.predict(X))


def test_reloaded_exposes_the_same_diagnostics():
    m = _fit(TrepanReloadedClassifier)
    assert m.stop_summary_["nodes_after_pruning"] == m.node_count_ and m.node_audit_
    row = m.semantic_split_audit_
    assert isinstance(row, list)


def test_extractor_audit_carries_node_audit_and_stop_summary():
    from sklearn.neural_network import MLPClassifier
    from core.trepan_original import TrepanOriginalExtractor
    X = np.random.default_rng(2).normal(size=(200, 4))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    mlp = MLPClassifier((8,), max_iter=300, random_state=0).fit(X, y)
    ex = TrepanOriginalExtractor(random_state=1)
    ex.extract_tree(mlp, X, y, sample_size=300, feature_names=list("abcd"), class_names=["0", "1"])
    audit = ex.last_audit
    assert audit["stop_summary"]["nodes_after_pruning"] == audit["node_count"]
    assert audit["node_audit"] and "loop_end_reason" in audit["stop_summary"]
