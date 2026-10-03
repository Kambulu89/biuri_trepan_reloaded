"""O resumo de paragem para a interface deriva do que o TREPAN já regista (só leitura)."""
import numpy as np

from core.tree_stop_summary import stop_summary_of
from core.trepan_original import TrepanOriginalClassifier


class Oracle:
    classes_ = np.array([0, 1])

    def __init__(self, fn):
        self.fn = fn

    def predict(self, X):
        return self.fn(np.asarray(X, float)).astype(int)


TWO_OF_THREE = Oracle(lambda X: (X[:, :3] > 0).sum(axis=1) >= 2)
CONSTANT = Oracle(lambda X: np.zeros(len(X)))


def _fit(oracle=TWO_OF_THREE, **over):
    X = np.random.default_rng(1).normal(size=(260, 5))
    kw = dict(max_nodes=11, max_depth=4, min_sample=180, max_queries=2200, max_n=3, beam_width=2, random_state=1)
    kw.update(over)
    return TrepanOriginalClassifier(**kw).fit(X, oracle=oracle, feature_names=list("abcde"))


def test_summary_fields_are_consistent_with_the_model():
    m = _fit()
    s = stop_summary_of(m)
    assert {"loop_end_reason", "node_budget", "nodes_before_pruning", "nodes_after_pruning", "query_budget",
            "queries_used", "query_budget_exhausted", "max_depth", "stop_reasons"} <= set(s)
    assert s["nodes_after_pruning"] == m.node_count_ <= s["nodes_before_pruning"] <= s["node_budget"]
    assert s["queries_used"] == m.membership_queries_ and s["query_budget"] == 2200
    assert sum(s["stop_reasons"].values()) >= 1


def test_node_budget_and_pure_oracle_reasons():
    assert stop_summary_of(_fit(max_nodes=3))["loop_end_reason"] == "node_budget_exhausted"
    pure = stop_summary_of(_fit(oracle=CONSTANT))
    assert pure["loop_end_reason"] == "no_expandable_nodes_left" and "STOP_PURE_NODE" in pure["stop_reasons"]


def test_unfitted_or_missing_model_gives_empty_summary_and_is_read_only():
    assert stop_summary_of(None) == {}
    m = _fit()
    rules = m.export_rules()
    stop_summary_of(m)
    assert m.export_rules() == rules
