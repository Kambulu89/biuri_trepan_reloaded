from __future__ import annotations

import numpy as np

from core.controlled_trepan_experiment import ControlledTrepanConfig
from core.trepan_original import ConstraintSet, MofNTest, Literal, _Node
from core.trepan_reloaded_historical import TrepanReloadedClassifier


class RuleOracle:
    """Oráculo determinístico com probabilidades, sem depender de dataset conhecido."""
    classes_ = np.asarray([0, 1])

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        # margem suave em torno de uma regra 2-of-3
        votes = (X[:, :3] > 0).sum(axis=1).astype(float)
        margin = (votes - 1.5) * 2.2
        p1 = 1.0 / (1.0 + np.exp(-margin))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _configured_model() -> TrepanReloadedClassifier:
    model = TrepanReloadedClassifier(
        max_nodes=7,
        max_depth=3,
        min_sample=120,
        max_queries=300,
        max_n=3,
        random_state=17,
        error_focused_refinement=True,
        error_focus_min_disagreement=0.05,
        error_focus_min_local_fidelity_gain=0.01,
    )
    model.classes_ = np.asarray([0, 1])
    model.n_features_in_ = 4
    model.feature_names_in_ = ["a", "b", "c", "d"]
    model.semantic_feature_weights_ = np.ones(4, dtype=float)
    model.semantic_feature_groups_ = ["signal", "signal", "signal", None]
    rel = np.eye(4)
    rel[:3, :3] = 0.9
    np.fill_diagonal(rel, 1.0)
    model.semantic_relatedness_matrix_ = rel
    return model


def test_error_region_profile_detects_disagreement_and_semantic_opportunity():
    model = _configured_model()
    X = np.asarray([
        [-2.0, -1.0, -1.0, 0.0],
        [-1.5, -0.8, -0.5, 0.2],
        [1.8, 1.2, -0.2, 0.1],
        [2.0, 1.4, 0.8, 0.0],
        [2.2, 1.1, 1.0, -0.1],
        [1.6, 0.9, 0.7, 0.3],
    ])
    y = np.asarray([0, 0, 1, 1, 1, 1])
    node = _Node(
        X, y, 1, ConstraintSet(), 0.55,
        np.asarray([0.5, 0.5]), 0, node_id=4,
    )
    profile = model._build_error_region_profile(node, X, y, include_uncertainty=False)
    assert profile["node_id"] == 4
    assert np.isclose(profile["disagreement_rate"], 4 / 6)
    assert profile["semantic_opportunity"] > 0.0
    scores = np.asarray(profile["semantic_feature_scores"])
    assert scores.shape == (4,)
    assert np.max(scores[:3]) > scores[3]


def test_local_fidelity_gate_rejects_semantic_candidate_that_does_not_improve():
    model = _configured_model()
    # y é perfeitamente separado por feature 0; feature 1 é uma proposta semântica pior.
    X = np.asarray([
        [-2.0, -0.1], [-1.0, 0.2], [-0.8, 1.0],
        [0.8, -1.0], [1.2, 0.1], [2.0, -0.2],
    ])
    y = np.asarray([0, 0, 0, 1, 1, 1])
    data_test = MofNTest(1, (Literal(0, 0.0, True),))
    semantic_test = MofNTest(1, (Literal(1, 0.0, True),))
    decision = model._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test,
        disagreement_rate=0.5,
    )
    assert decision["attempted"] is True
    assert decision["accepted"] is False
    assert decision["semantic_local_fidelity"] < decision["data_only_local_fidelity"]
    assert decision["reason"] == "semantic_candidate_no_local_fidelity_gain"


def test_local_fidelity_gate_accepts_semantic_candidate_with_real_gain():
    model = _configured_model()
    X = np.asarray([
        [-2.0, -2.0], [-1.0, -1.5], [-0.8, -1.0],
        [0.8, 1.0], [1.2, 1.5], [2.0, 2.0],
    ])
    y = np.asarray([0, 0, 0, 1, 1, 1])
    data_test = MofNTest(1, (Literal(0, 10.0, True),))  # quase inútil
    semantic_test = MofNTest(1, (Literal(1, 0.0, True),))
    decision = model._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test,
        disagreement_rate=0.5,
    )
    assert decision["attempted"] is True
    assert decision["accepted"] is True
    assert decision["local_fidelity_gain"] >= model.error_focus_min_local_fidelity_gain


def test_error_focused_membership_queries_stay_within_same_budget_and_are_audited():
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 5))
    oracle = RuleOracle()
    rel = np.eye(5)
    rel[:3, :3] = 0.9
    np.fill_diagonal(rel, 1.0)
    model = TrepanReloadedClassifier(
        max_nodes=7,
        max_depth=3,
        min_sample=180,
        max_queries=240,
        max_n=3,
        random_state=7,
        semantic_active_query_fraction=0.75,
        semantic_active_pool_multiplier=4,
        error_focused_refinement=True,
        error_focus_min_disagreement=0.01,
    ).fit(
        X,
        oracle=oracle,
        feature_names=[f"f{i}" for i in range(5)],
        semantic_feature_weights=np.ones(5),
        semantic_feature_groups=["g", "g", "g", None, None],
        semantic_relatedness_matrix=rel,
    )
    assert model.membership_queries_ <= 240
    summary = model.semantic_audit_summary_
    assert summary["error_regions_evaluated"] > 0
    assert summary["error_focused_query_batches"] > 0
    assert summary["error_focused_query_selected"] > 0
    assert summary["probability_rows_reused_for_uncertainty"] >= 0
    assert model.oracle_ is None


def test_controlled_config_keeps_error_focus_reload_only_not_common_budget():
    cfg = ControlledTrepanConfig()
    common = cfg.common_tree_kwargs()
    assert cfg.error_focused_refinement is True
    assert "error_focused_refinement" not in common
    assert "error_focus_strength" not in common
    assert common["max_queries"] == cfg.max_queries


def test_low_disagreement_region_does_not_allow_semantic_replacement():
    model = _configured_model()
    X = np.asarray([[-2.0, -2.0], [-1.0, -1.0], [1.0, 1.0], [2.0, 2.0]])
    y = np.asarray([0, 0, 1, 1])
    data_test = MofNTest(1, (Literal(0, 0.0, True),))
    semantic_test = MofNTest(1, (Literal(1, 0.0, True),))
    decision = model._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test,
        disagreement_rate=0.0,
    )
    assert decision["attempted"] is False
    assert decision["accepted"] is False
    assert decision["reason"] == "region_below_disagreement_threshold"


def test_efsr_can_turn_semantic_error_structure_into_measurable_fidelity_gain():
    from core.trepan_original import TrepanOriginalClassifier

    rng = np.random.default_rng(123)
    n = 600
    signal = rng.integers(0, 2, size=(n, 3)).astype(float)
    y = (signal.sum(axis=1) >= 2).astype(int)
    distractor = (y ^ (rng.random(n) < 0.10)).astype(float)
    noise = rng.normal(size=n)
    X = np.column_stack([signal, distractor, noise])

    class MajorityOracle:
        classes_ = np.asarray([0, 1])

        def predict(self, values):
            values = np.asarray(values, dtype=float)
            return (values[:, :3].sum(axis=1) >= 1.5).astype(int)

        def predict_proba(self, values):
            pred = self.predict(values)
            p1 = np.where(pred == 1, 0.92, 0.08)
            return np.column_stack([1.0 - p1, p1])

    oracle = MajorityOracle()
    common = dict(
        max_nodes=3,
        max_depth=1,
        min_sample=n,
        max_queries=0,
        max_n=3,
        beam_width=1,
        max_features_per_node=5,
        min_samples_leaf=4,
        random_state=9,
    )
    names = ["a", "b", "c", "distractor", "noise"]
    original = TrepanOriginalClassifier(**common).fit(X, oracle=oracle, feature_names=names)

    rel = np.eye(5)
    rel[:3, :3] = 0.95
    np.fill_diagonal(rel, 1.0)
    reloaded = TrepanReloadedClassifier(
        **common,
        semantic_candidate_budget=80,
        error_focus_min_disagreement=0.01,
        error_focus_min_local_fidelity_gain=0.001,
    ).fit(
        X,
        oracle=oracle,
        feature_names=names,
        semantic_feature_weights=np.ones(5),
        semantic_feature_groups=["signal", "signal", "signal", None, None],
        semantic_relatedness_matrix=rel,
    )
    teacher = oracle.predict(X)
    fid_original = float(np.mean(original.predict(X) == teacher))
    fid_reloaded = float(np.mean(reloaded.predict(X) == teacher))
    assert fid_reloaded > fid_original
    assert fid_reloaded == 1.0
    row = reloaded.semantic_split_audit_[0]
    assert row["error_focused_intervention_accepted"] is True
    assert row["local_fidelity_gain"] > 0.05
    assert set(row["semantic_features"]) >= {0, 1, 2}
