"""Contratos científicos da construção de árvores: C4.5, TREPAN Original e Reloaded.

Estes testes falham se o TREPAN deixar de consultar o oráculo, passar a usar
rótulos reais, deixar de ser best-first, deixar de avaliar m-of-n, ignorar o
orçamento de queries, violar as restrições do caminho, esconder motivos de
paragem, usar o teste para podar, ou se o C4.5 passar a depender de
``DecisionTreeClassifier``.
"""
from __future__ import annotations

import inspect
import pickle
import re
from pathlib import Path

import numpy as np
import pytest

from core import c45_j48_tree
from core.c45_j48_tree import C45Classifier
from core.trepan_original import (
    Literal, MofNTest, StopReason, TrepanOriginalClassifier, TrepanOriginalExtractor,
)
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.tree_build_report import (
    c45_build_report, format_report, stump_diagnostic, tree_cache_key, trepan_build_report,
)

ROOT = Path(__file__).resolve().parents[1]


class CountingOracle:
    """Oráculo 2-of-3 sobre as primeiras features; conta consultas."""

    def __init__(self, noise_features: int = 0):
        self.calls = 0
        self.rows = 0

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        self.calls += 1
        self.rows += len(X)
        return ((X[:, 0] > 0).astype(int) + (X[:, 1] > 0) + (X[:, 2] > 0) >= 2).astype(int)


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 6))
    y_real = (X[:, 0] + 0.5 * rng.normal(size=300) > 0).astype(int)  # distinto do oráculo
    return X, y_real


def build(X, oracle=None, **kw):
    params = dict(min_sample=500, max_queries=50000, max_nodes=15, max_depth=5, random_state=7)
    params.update(kw)
    tree = TrepanOriginalClassifier(**params)
    tree.fit(X, oracle=oracle or CountingOracle(), feature_names=[f"f{i}" for i in range(X.shape[1])])
    return tree


# ------------------------------------------------------------------ TREPAN Original
def test_trepan_queries_oracle_and_ignores_real_labels(data):
    X, y_real = data
    oracle = CountingOracle()
    a = build(X, oracle)
    assert oracle.calls > 1 and a.membership_queries_ > 0, "TREPAN deve fazer membership queries"
    # y real passado por engano não pode influenciar a árvore
    b = TrepanOriginalClassifier(min_sample=500, max_queries=50000, max_nodes=15, max_depth=5, random_state=7)
    b.fit(X, y=y_real, oracle=CountingOracle(), feature_names=[f"f{i}" for i in range(6)])
    assert np.array_equal(a.predict(X), b.predict(X))
    assert "y_test" not in inspect.signature(TrepanOriginalClassifier.fit).parameters
    assert not any(p.startswith("X_test") for p in inspect.signature(TrepanOriginalClassifier.fit).parameters)


def test_query_budget_respected_and_reported(data):
    X, _ = data
    t = build(X, max_queries=450, min_sample=1000)
    assert t.membership_queries_ <= 450
    assert t.query_budget_exhausted_ is True
    codes = {n.stop_reason for n in t.iter_nodes() if n.is_leaf}
    assert StopReason.QUERY_BUDGET_EXHAUSTED in codes or t.membership_queries_ >= 450
    rep = trepan_build_report(t, algorithm="TREPAN Original", oracle_name="MLP Original")
    c = rep["CONSTRUCTION"]
    assert c["query_budget_initial"] == 450 and c["query_budget_used"] == t.membership_queries_
    assert c["query_budget_remaining"] == 450 - t.membership_queries_


def test_budget_not_exhausted_when_large(data):
    X, _ = data
    t = build(X, max_queries=500000)
    assert t.query_budget_exhausted_ is False
    assert StopReason.QUERY_BUDGET_EXHAUSTED not in {n.stop_reason for n in t.iter_nodes()}


def test_queries_respect_path_constraints(data):
    X, _ = data
    t = build(X)
    checked = 0
    for node in t.iter_nodes():
        if len(node.query_X):
            assert node.constraints.accepts(node.query_X).all()
            assert node.stats["queries_violating_path"] == 0
            checked += len(node.query_X)
    assert checked > 0


def test_node_effective_sample_augmented_by_queries(data):
    X, _ = data
    t = build(X, min_sample=450)
    root = t.root_
    assert root.stats["real_samples"] == len(X)
    assert root.stats["effective_samples"] >= 450
    assert root.stats["synthetic_samples"] == root.stats["queries_used"]


def test_best_first_is_real(data):
    X, _ = data
    t = build(X, max_nodes=21)
    log = t.expansion_log_
    assert len(log) >= 2
    assert all(item["is_best_first_choice"] for item in log)
    assert [item["selected_order"] for item in log] == list(range(1, len(log) + 1))
    for key in ("candidate_node_id", "priority_score", "samples", "impurity", "potential_fidelity_gain",
                "priority_components"):
        assert key in log[0]


def test_m_of_n_is_evaluated_and_can_be_selected(data):
    X, _ = data
    t = build(X)
    rep = trepan_build_report(t, algorithm="TREPAN Original", oracle_name="O")
    c = rep["CONSTRUCTION"]
    assert c["simple_candidates_evaluated"] > 0
    assert c["m_of_n_candidates_evaluated"] > 0
    assert c["m_of_n_split_wins"] >= 1, "o oráculo 2-of-3 deve levar à selecção de um m-of-n"
    assert c["simple_split_wins"] + c["m_of_n_split_wins"] == len(t.split_audit_)
    assert t.m_of_n_ is True


def test_m_of_n_prediction_engine():
    # 2-of-3: A > 1, B <= 5, C > 0.5
    test = MofNTest(2, (Literal(0, 1.0, True), Literal(1, 5.0, False), Literal(2, 0.5, True)))
    X = np.array([
        [2.0, 4.0, 0.0],   # A, B verdadeiros -> 2 -> True
        [0.0, 9.0, 0.0],   # nenhum           -> False
        [2.0, 9.0, 0.0],   # só A             -> False
        [0.0, 4.0, 1.0],   # B, C             -> True
        [2.0, 4.0, 1.0],   # três             -> True
    ])
    assert test.evaluate(X).tolist() == [True, False, False, True, True]
    # <= e > em fronteira
    assert Literal(0, 1.0, True).evaluate(np.array([[1.0]])).tolist() == [False]
    assert Literal(0, 1.0, False).evaluate(np.array([[1.0]])).tolist() == [True]


def test_every_leaf_has_explicit_stop_reason(data):
    X, _ = data
    t = build(X)
    for node in t.iter_nodes():
        if node.is_leaf:
            assert node.stop_reason in StopReason.ALL, node.node_id


def test_stop_max_depth_max_nodes_min_gain(data):
    X, _ = data
    d = build(X, max_depth=1)
    assert d.get_depth() == 1
    assert {n.stop_reason for n in d.iter_nodes() if n.is_leaf} == {StopReason.MAX_DEPTH}
    m = build(X, max_nodes=3)
    assert StopReason.MAX_NODES in {n.stop_reason for n in m.iter_nodes() if n.is_leaf}
    g = build(X, min_gain=5.0)  # impossível: o IG binário é <= 1
    root = g.root_
    assert root.is_leaf and root.stop_reason == StopReason.MIN_GAIN
    assert root.stop_detail["required_min_gain"] == 5.0
    assert 0.0 <= root.stop_detail["best_candidate_gain"] < 5.0


def test_pure_oracle_stops_with_pure_reason(data):
    X, _ = data

    class Const:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)

    t = TrepanOriginalClassifier(min_sample=50, max_queries=1000, random_state=1).fit(X, oracle=Const())
    assert t.root_.is_leaf and t.root_.stop_reason == StopReason.PURE_NODE


def test_stump_diagnostic_explains_tree(data):
    X, _ = data
    t = build(X, max_nodes=1)
    diag = stump_diagnostic(t)
    assert diag["alert"] == "TREE_STUMP_DIAGNOSTIC" and diag["max_nodes_reached"] is True
    rep = trepan_build_report(t, algorithm="TREPAN Original", oracle_name="O")
    assert "TREE_STUMP_DIAGNOSTIC" in format_report(rep)
    big = build(X, max_nodes=15)
    if big.node_count_ > 3:
        assert stump_diagnostic(big) is None


def test_raw_tree_preserved_and_pruning_audited(data):
    X, _ = data
    t = build(X, max_nodes=31, max_depth=6)
    s = t.pruning_summary_
    assert s["nodes_before_pruning"] == t.nodes_raw_ >= s["nodes_after_pruning"] == t.node_count_
    assert s["uses_test_data"] is False
    assert t.tree_raw_root_ is not t.root_
    assert len(t.predict_raw(X)) == len(X)
    assert s["pruned_nodes"] == s["nodes_before_pruning"] - s["nodes_after_pruning"]
    for entry in t.pruning_audit_:
        assert entry["uses_test_data"] is False and entry["reason"]


def test_fidelity_vs_oracle_and_accuracy_vs_real_are_separate(data):
    X, y_real = data
    oracle = CountingOracle()
    t = build(X, oracle)
    rep = trepan_build_report(
        t, algorithm="TREPAN Original", oracle_name="MLP Original", X_eval=X,
        y_oracle_eval=oracle.predict(X), y_real_eval=y_real,
    )
    m = rep["METRICS"]
    assert m["fidelity"] == pytest.approx(float(np.mean(t.predict(X) == oracle.predict(X))))
    assert m["accuracy_real"] == pytest.approx(float(np.mean(t.predict(X) == y_real)))
    assert m["fidelity"] != m["accuracy_real"]
    assert m["fidelity_target"] == "MLP Original"


def test_deterministic_given_seed(data):
    X, _ = data
    a, b = build(X, random_state=3), build(X, random_state=3)
    assert np.array_equal(a.predict(X), b.predict(X))
    assert [s["test"] for s in a.split_audit_] == [s["test"] for s in b.split_audit_]
    assert a.membership_queries_ == b.membership_queries_
    assert [e["candidate_node_id"] for e in a.expansion_log_] == [e["candidate_node_id"] for e in b.expansion_log_]


def test_serialization_roundtrip_keeps_audit(data):
    X, _ = data
    t = build(X)
    t2 = pickle.loads(pickle.dumps(t))
    assert np.array_equal(t.predict(X), t2.predict(X))
    assert [n.stop_reason for n in t.iter_nodes()] == [n.stop_reason for n in t2.iter_nodes()]
    assert t2.pruning_summary_ == t.pruning_summary_
    assert len(t2.predict_raw(X)) == len(X)
    assert [s["test"] for s in t.split_audit_] == [s["test"] for s in t2.split_audit_]


def test_extractor_ignores_test_set_for_tree_construction(data):
    X, y_real = data

    class FakeMLP(CountingOracle):
        pass

    def run(**kw):
        ex = TrepanOriginalExtractor(random_state=5)
        ex.extract_tree(FakeMLP(), X[:200], y_real[:200], sample_size=20000, X_train=X[:200],
                        training_limits={"min_sample": 500, "max_nodes": 15}, **kw)
        return ex

    a = run()
    b = run(X_test=X[200:], y_test=y_real[200:])
    assert np.array_equal(a.explainer_tree.predict(X), b.explainer_tree.predict(X))
    audit = b.last_audit
    assert audit["final_test_used_for_selection"] is False
    assert audit["oracle_name"] == "MLP Original"
    assert "trepan_fidelity" in audit and "trepan_accuracy" in audit
    assert audit["build_report"]["METRICS"]["fidelity_target"] == "MLP Original"


# ------------------------------------------------------------------ Reloaded
def test_reloaded_inherits_trepan_foundation():
    assert issubclass(TrepanReloadedClassifier, TrepanOriginalClassifier)


def test_reloaded_semantics_off_matches_original_and_has_no_bonus(data):
    X, _ = data
    names = [f"f{i}" for i in range(6)]
    kw = dict(min_sample=500, max_queries=50000, max_nodes=15, max_depth=5, random_state=7)
    orig = TrepanOriginalClassifier(**kw).fit(X, oracle=CountingOracle(), feature_names=names)
    rel = TrepanReloadedClassifier(alpha=0.0, beta=0.0, **kw).fit(X, oracle=CountingOracle(), feature_names=names)
    assert np.array_equal(orig.predict(X), rel.predict(X))
    for row in rel.semantic_split_audit_:
        assert row["semantic_bonus"] == pytest.approx(0.0)
        assert row["selection_score"] == pytest.approx(row["information_gain"])
        assert row["ontology_influenced"] is False
    # relatório de construção partilhado pela base TREPAN
    rep = trepan_build_report(rel, algorithm="TREPAN Reloaded", oracle_name="MLP Original")
    assert rep["CONSTRUCTION"]["best_first_verified"] is True


def test_reloaded_semantic_bonus_cannot_rescue_zero_information_split():
    tree = TrepanReloadedClassifier(alpha=1.0, beta=1.0)
    tree.classes_ = np.array([0, 1])
    y = np.array([0, 1] * 20)
    mask = np.array([True, True, False, False] * 10)  # y alterna: partição sem informação
    assert tree._split_selection_score(y, mask, None) == pytest.approx(0.0)


# ------------------------------------------------------------------ C4.5
def test_c45_is_native_gain_ratio_not_cart_entropy():
    src = Path(c45_j48_tree.__file__).read_text(encoding="utf-8")
    assert "DecisionTreeClassifier" not in re.sub(r'""".*?"""', "", src, flags=re.S).replace(
        "``DecisionTreeClassifier``", "")
    assert C45Classifier.is_c45_native is True
    assert "gain_ratio" in src and "min_gain_ratio" in inspect.signature(C45Classifier.__init__).parameters


def test_c45_gain_ratio_prefers_binary_attribute_over_id_like_attribute():
    # f0: identificador (ganho máximo, split-info alto); f1: binário quase perfeito; f2: ruído.
    X = np.tile(np.array([[i, 0 if i < 4 else 1, i % 2] for i in range(8)], dtype=float), (10, 1))
    y = np.tile(np.array([0, 0, 0, 1, 1, 1, 1, 1]), 10)
    tree = C45Classifier(min_samples_split=2, min_samples_leaf=1, feature_types=["categorical", "numeric", "numeric"])
    tree.fit(X, y)
    assert tree.root_.feature == 1, "Gain Ratio deve preferir f1 (IG puro escolheria o ID)"
    assert tree.root_.gain_ratio > 0


def test_c45_uses_only_real_labels_and_reports_audit(data):
    X, y_real = data
    assert "oracle" not in inspect.signature(C45Classifier.fit).parameters
    tree = C45Classifier().fit(X, y_real)
    rep = c45_build_report(tree, X_eval=X, y_real_eval=y_real)
    assert rep["uses_oracle"] is False and rep["uses_membership_queries"] is False
    assert rep["split_criterion"] == "gain_ratio"
    assert rep["STRUCTURE"]["nodes_raw"] >= rep["STRUCTURE"]["nodes_final"]
    assert rep["PRUNING"]["uses_test_data"] is False
    assert set(rep["STOP_REASONS"]) <= set(StopReason.ALL)
    assert "fidelity" not in rep["METRICS"]
    assert rep["METRICS"]["accuracy_real"] == pytest.approx(float(np.mean(tree.predict(X) == y_real)))


# ------------------------------------------------------------------ cache
def test_tree_cache_key_changes_with_every_relevant_parameter():
    base = dict(algorithm="trepan_original", dataset_hash="d", oracle_hash="o", ontology_hash="x",
                parameters=dict(query_budget=10000, max_depth=8, max_nodes=31, min_samples=4,
                                min_gain=0.0, semantic_weight=0.0, pruning_config="identical",
                                seed=42))
    key = tree_cache_key(**base)
    assert key == tree_cache_key(**base)
    for field in ("algorithm", "dataset_hash", "oracle_hash", "ontology_hash"):
        assert tree_cache_key(**{**base, field: "changed"}) != key
    for name, value in base["parameters"].items():
        params = dict(base["parameters"]); params[name] = "changed"
        assert tree_cache_key(**{**base, "parameters": params}) != key, name
    assert tree_cache_key(**{**base, "code_version": "other"}) != key
