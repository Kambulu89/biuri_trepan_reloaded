from __future__ import annotations

import numpy as np

from core.trepan_original import ConstraintSet, Literal, MofNTest, TrepanOriginalClassifier, _Node
from core.trepan_reloaded_historical import TrepanReloadedClassifier
from core.trepan_reloaded_extractor import TrepanReloadedExtractor


class _Entity:
    def __init__(self, name, *, parents=None):
        self.name = name
        self.iri = f"urn:test:{name}"
        self.is_a = list(parents or [])
        self.equivalent_to = []
        self.domain = []
        self.range = []
        self.inverse_property = []
        self.label = []

    def ancestors(self):
        out = {self}
        stack = list(self.is_a)
        while stack:
            item = stack.pop()
            if item in out:
                continue
            out.add(item)
            stack.extend(getattr(item, "is_a", []) or [])
        return out


class _Ontology:
    def __init__(self):
        morphology = _Entity("Morphology")
        size = _Entity("SizeMorphometry", parents=[morphology])
        contour = _Entity("ContourIrregularity", parents=[morphology])
        self._classes = [morphology]
        self._props = [
            _Entity("hasRadius", parents=[size]),
            _Entity("hasArea", parents=[size]),
            _Entity("hasConcavity", parents=[contour]),
        ]

    def classes(self): return list(self._classes)
    def data_properties(self): return list(self._props)
    def object_properties(self): return []
    def annotation_properties(self): return []


def _matches():
    return [
        {"feature": "radius", "entity_name": "hasRadius", "entity_type": "datatype_property", "score": 1.0, "accepted": True},
        {"feature": "area", "entity_name": "hasArea", "entity_type": "datatype_property", "score": 1.0, "accepted": True},
        {"feature": "concavity", "entity_name": "hasConcavity", "entity_type": "datatype_property", "score": 1.0, "accepted": True},
    ]


def test_quality_matches_are_promoted_to_semantic_graph_relations_and_groups():
    extractor = TrepanReloadedExtractor(ontology=_Ontology())
    extractor.ontology_quality_report = {
        "accepted": True,
        "matches": _matches(),
        "metrics": {"total_features": 3, "mapped_features": 3, "feature_coverage": 1.0},
    }
    extractor.register_quality_matches(["radius", "area", "concavity"], _matches())

    summary = getattr(extractor, "semantic_graph_summary", {})
    assert summary.get("mapped_features") == 3
    assert summary.get("edges", 0) > 0
    assert summary.get("features_with_groups", 0) >= 3
    assert extractor.feature_semantics[0].get("semantic_group") == "SizeMorphometry"
    assert extractor.feature_semantics[1].get("semantic_group") == "SizeMorphometry"
    assert extractor.feature_semantics[2].get("semantic_group") == "ContourIrregularity"
    assert len(extractor.domain_knowledge.get("relationships", [])) > 0


def test_error_region_topk_fallback_makes_real_error_eligible_below_fixed_threshold():
    model = TrepanReloadedClassifier(
        max_nodes=3, max_depth=1, min_sample=20, max_queries=0,
        error_focused_refinement=True,
        error_focus_min_disagreement=0.20,
        error_focus_top_k=2,
        error_focus_min_regions=1,
        random_state=1,
    )
    model.classes_ = np.asarray([0, 1])
    model.n_features_in_ = 3
    model.feature_names_in_ = ["a", "b", "c"]
    model.semantic_feature_weights_ = np.ones(3)
    model.semantic_feature_groups_ = ["g", "g", None]
    rel = np.eye(3); rel[0, 1] = rel[1, 0] = 0.9
    model.semantic_relatedness_matrix_ = rel
    model._error_focus_fallback_used_ = 0

    X = np.asarray([
        [-2.0, -1.0, 0.0], [-1.0, -0.5, 0.1], [1.0, 0.8, 0.2],
        [1.2, 0.9, -0.1], [1.5, 1.0, 0.0], [1.7, 1.1, 0.2],
        [1.8, 1.2, 0.1], [2.0, 1.3, 0.0], [2.1, 1.4, -0.2], [2.2, 1.5, 0.0],
    ])
    # current prediction 1 -> only one disagreement = 10%, below 20%
    y = np.asarray([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    node = _Node(X, y, 0, ConstraintSet(), 1.0, np.asarray([0.1, 0.9]), 1, node_id=0)
    profile = model._build_error_region_profile(node, X, y, include_uncertainty=False)
    eligible, reason = model._resolve_error_region_eligibility(profile, consume=True)
    assert profile["disagreement_rate"] < model.error_focus_min_disagreement
    assert eligible is True
    assert reason == "top_k_error_region_fallback"


def test_efsr_rejects_candidate_that_improves_queries_but_hurts_real_training_rows():
    model = TrepanReloadedClassifier(
        error_focused_refinement=True,
        error_focus_min_disagreement=0.01,
        error_focus_min_local_fidelity_gain=0.01,
        error_focus_min_real_fidelity_gain=0.0,
    )
    model.classes_ = np.asarray([0, 1])

    real_X = np.asarray([
        [-2.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [2.0, 1.0],
    ])
    real_y = np.asarray([0, 0, 1, 1])
    # Synthetic/query rows favour feature 1 strongly and make feature 0 noisy.
    syn_X = np.asarray([
        [2.0, -2.0], [2.2, -1.5], [-2.0, 1.5], [-2.2, 2.0],
        [1.5, -1.0], [-1.5, 1.0],
    ])
    syn_y = np.asarray([0, 0, 1, 1, 0, 1])
    X = np.vstack([real_X, syn_X])
    y = np.concatenate([real_y, syn_y])
    data_test = MofNTest(1, (Literal(0, 0.0, True),))
    semantic_test = MofNTest(1, (Literal(1, 0.0, True),))
    model._current_real_validation_X_ = real_X
    model._current_real_validation_y_ = real_y

    decision = model._evaluate_error_focused_intervention(
        X, y, data_test=data_test, semantic_test=semantic_test, disagreement_rate=0.5,
    )
    assert decision["semantic_local_fidelity"] > decision["data_only_local_fidelity"]
    assert decision["semantic_real_fidelity"] < decision["data_only_real_fidelity"]
    assert decision["accepted"] is False
    assert decision["reason"] == "semantic_candidate_fails_real_training_gate"


class _SoftOracle:
    classes_ = np.asarray([0, 1])
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return (X[:, 0] + 0.6 * X[:, 1] > 0).astype(int)
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        z = X[:, 0] + 0.6 * X[:, 1]
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1-p, p])


def test_no_measurable_semantic_effect_mirrors_exact_original_tree():
    rng = np.random.default_rng(8)
    X = rng.normal(size=(180, 3))
    oracle = _SoftOracle()
    common = dict(
        max_nodes=7, max_depth=3, min_sample=220, max_queries=160,
        max_n=2, beam_width=2, max_features_per_node=3, random_state=8,
    )
    original = TrepanOriginalClassifier(**common).fit(X, oracle=oracle, feature_names=["a", "b", "c"])

    # Structural semantics are present, but strength zero means they cannot earn
    # a measurable accepted intervention. Query sampling may otherwise diverge.
    reloaded = TrepanReloadedClassifier(
        **common,
        semantic_gain_strength=0.0,
        semantic_group_strength=0.0,
        semantic_active_query_fraction=0.9,
        error_focused_refinement=True,
        error_focus_min_local_fidelity_gain=0.5,
        mirror_when_no_semantic_effect=True,
    ).fit(
        X, oracle=oracle, feature_names=["a", "b", "c"],
        semantic_feature_weights=np.ones(3),
        semantic_feature_groups=["g", "g", None],
        semantic_relatedness_matrix=np.asarray([[1,.9,0],[.9,1,0],[0,0,1]], dtype=float),
    )
    assert np.array_equal(reloaded.predict(X), original.predict(X))
    assert reloaded.export_rules() == original.export_rules()
    assert getattr(reloaded, "semantic_effect_mirror_applied_", False) is True
