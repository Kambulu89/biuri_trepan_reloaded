"""Modelo de dados da experiência: valores em falta com razão, mapeamento do backend, build info."""
import json

import numpy as np
import pytest

from core.build_info import build_label, get_build_info
from core.experiment_builders import (
    enrichment_from_reports, ontology_from_quality, predictive_metrics, semantic_features_from_audit,
    semantic_splits_from_audit, tree_diagnostics,
)
from core.experiment_result import (
    ExperimentResult, ExperimentState, Measure, ModelCard, Provenance, Reason, to_jsonable,
)


# ------------------------------------------------------------------------------ Measure ---
@pytest.mark.parametrize("raw", [None, float("nan"), float("inf"), "abc", object()])
def test_measure_never_turns_missing_into_zero(raw):
    m = Measure.of(raw, Reason.NO_ORACLE)
    assert m.value is None and m.reason == Reason.NO_ORACLE and not m.available


def test_measure_keeps_real_zero_distinct_from_missing():
    zero = Measure.of(0.0)
    assert zero.value == 0.0 and zero.available and zero.reason is None
    assert Measure.of(np.float64(0.25)).value == 0.25


def test_every_reason_has_a_category():
    codes = [v for k, v in vars(Reason).items() if k.isupper() and isinstance(v, str)]
    assert codes and all(c in Reason.CATEGORY for c in codes)


def test_to_jsonable_handles_measure_numpy_enum_and_nan():
    obj = {"m": Measure.of(1.5), "na": Measure.na(Reason.NOT_TRAINED), "a": np.arange(2), "s": np.float32(2.5),
           "e": ExperimentState.NO_DATA, "n": float("nan")}
    out = to_jsonable(obj)
    assert out["m"] == {"value": 1.5, "reason": None} and out["na"] == {"value": None, "reason": "not_trained"}
    assert out["a"] == [0, 1] and out["s"] == 2.5 and out["e"] == "NO_DATA" and out["n"] is None
    json.dumps(out)


def test_empty_experiment_result_is_valid_and_serializable():
    r = ExperimentResult()
    d = r.to_dict()
    assert d["state"] == "NO_DATA" and d["schema_version"] == "1.0" and d["provenance"]["experiment_id"]
    json.dumps(d)
    assert Provenance().experiment_id != Provenance().experiment_id  # ids únicos


# ----------------------------------------------------------------------------- build info ---
def test_build_info_is_complete_and_never_fails(monkeypatch):
    info = get_build_info()
    assert {"version", "commit", "dirty", "semantic_pipeline_version", "python"} <= set(info)
    assert info["version"] != "unknown" and info["semantic_pipeline_version"]
    label = build_label()
    assert info["version"] in label and info["commit"] in label


# ---------------------------------------------------------------------------- ontologia ---
QUALITY = {
    "accepted": True, "status": "VALID_DOMAIN_ONTOLOGY", "issues": [], "warnings": ["w1"],
    "metrics": {"feature_coverage": 1.0, "mapped_features": 30, "total_features": 30, "ambiguous_matches": 0,
                "entity_collisions": 0, "semantic_richness": {"level": "limited"},
                "knowledge_split": {"tbox": {"classes": 4, "datatype_properties": 30, "object_properties": 0}}},
    "abox": {"status": "SAFE", "accepted": True},
    "reasoner": {"executed": True, "consistent": True, "engine": "hermit", "duration_seconds": 0.5,
                 "inferred_axioms_count": 0},
}


def test_ontology_axes_are_separate():
    o = ontology_from_quality(QUALITY, path="x.owl", owl_hash="abc")
    assert (o.structural_status, o.reasoner_status, o.abox_status, o.tbox_status) == ("VALID", "CONSISTENT", "SAFE", "VALID")
    assert (o.mapped, o.total, o.coverage, o.ambiguous) == (30, 30, 1.0, 0) and o.richness == "limited"
    assert o.path == "x.owl" and o.hash == "abc" and o.reasoner_seconds == 0.5


def test_invalid_ontology_and_reasoner_not_executed():
    q = dict(QUALITY, accepted=False, status="GENERIC_ONTOLOGY", issues=["Ontologia sem TBox utilizável."],
             reasoner={"executed": False, "fallback": "explicit_axioms_only"})
    o = ontology_from_quality(q)
    assert o.structural_status == "GENERIC_ONTOLOGY" and o.reasoner_status == "NOT_EXECUTED"
    assert o.reasoner_fallback == "explicit_axioms_only" and o.tbox_status == "INVALID"


def test_inconsistent_reasoner_is_reported_as_inconsistent():
    q = dict(QUALITY, reasoner={"executed": True, "consistent": False})
    assert ontology_from_quality(q).reasoner_status == "INCONSISTENT"


def test_no_quality_report_means_not_evaluated_not_valid():
    o = ontology_from_quality(None, path="x.owl")
    assert o.loaded and o.structural_status == "NOT_EVALUATED" and o.abox_status == "NOT_EVALUATED"


# ------------------------------------------------------------------------ enriquecimento ---
REJECTED = {
    "decision": "REJECT_NO_INFORMATIONAL_GAIN", "semantic_mlp_accepted": False, "evidence_strength": "n/a",
    "selected_semantic_features": [],
    "stages": {"B_novelty": {"generated": 34}, "C_screening": {"stable_features": ["a", "b"]},
               "D_mlp_comparison": {"base": {"utility": 0.90}, "with_owl": {"utility": 0.905}, "utility_gain": 0.005,
                                    "utility_gain_ci": [-0.01, 0.02]}},
}


def test_rejected_enrichment_keeps_utilities_and_trepan_state_separate():
    e = enrichment_from_reports(REJECTED, {"semantic_trepan_available": True, "semantic_trepan_reason": "ok"},
                                {"teacher": "mlp_original"}, {"selected_mode": "neutral", "status": "NOT_ATTRIBUTED"})
    assert e.mlp_status == "REJECTED" and e.decision == "REJECT_NO_INFORMATIONAL_GAIN"
    assert (e.base_utility.value, e.owl_utility.value, e.delta_utility.value) == (0.90, 0.905, 0.005)
    assert e.utility_ci == [-0.01, 0.02] and (e.generated, e.stable, e.selected) == (34, 2, 0)
    assert e.trepan_semantics_available is True and e.reloaded_mode == "neutral"


def test_accepted_enrichment():
    rep = dict(REJECTED, decision="ACCEPT_PARTIAL_FEATURE_SET", semantic_mlp_accepted=True, evidence_strength="strong",
               selected_semantic_features=["a"])
    e = enrichment_from_reports(rep)
    assert e.mlp_status == "ACCEPTED" and e.selected == 1 and e.evidence_strength == "strong"


def test_not_evaluated_and_no_ontology_are_distinguished():
    assert enrichment_from_reports(None).mlp_status == "NOT_EVALUATED"
    none = enrichment_from_reports(None, ontology_loaded=False)
    assert none.mlp_status == "NOT_AVAILABLE" and none.trepan_semantics_available is False
    assert enrichment_from_reports(None).base_utility.reason == Reason.ENRICHMENT_NOT_EVALUATED


# ------------------------------------------------------------------------------- árvores ---
def test_tree_diagnostics_from_stop_summary():
    s = {"nodes_after_pruning": 5, "nodes_before_pruning": 7, "node_budget": 11, "query_budget": 2200,
         "queries_used": 900, "query_budget_exhausted": False, "loop_end_reason": "no_expandable_nodes_left",
         "stop_reasons": {"pure_node": 3}}
    d = tree_diagnostics("trepan_original", s, depth=2, leaves=3, split_audit=[{"n": 2}, {"n": 1}])
    assert d.available and d.logical_nodes.value == 5 and d.nodes_before_pruning.value == 7 and d.depth.value == 2
    assert d.queries_used.value == 900 and d.query_budget.value == 2200 and d.query_budget_exhausted is False
    assert d.stop_reasons == {"pure_node": 3} and d.m_of_n_splits == 1
    assert d.rendered_nodes.reason == Reason.NOT_RENDERED     # ainda não desenhada: razão explícita


def test_unbuilt_tree_is_not_executed_not_zero():
    d = tree_diagnostics("trepan_reloaded", None, built=False)
    assert not d.available and d.logical_nodes.reason == Reason.TREE_NOT_BUILT and d.depth.value is None


def test_predictive_metrics_missing_is_reasoned():
    m = predictive_metrics({"accuracy": 0.9, "f1_macro": 0.8}, missing=Reason.NOT_TRAINED)
    assert m["accuracy"].value == 0.9 and m["macro_f1"].value == 0.8      # alias f1_macro
    assert m["balanced_accuracy"].value is None and m["balanced_accuracy"].reason == Reason.NOT_TRAINED
    assert m["accuracy"].value != 0 and all(not v.available or v.value > 0 for v in m.values())


def test_semantic_tables_come_straight_from_audits():
    feats = semantic_features_from_audit({"feature_audit": [
        {"feature": "onto_x", "origin": "relational", "source_features": ["a", "b"], "selection_frequency": 0.8,
         "decision": "SELECTED", "reason": "stable_selection_across_folds"},
        {"feature": "onto_y", "origin": "aggregate", "source_features": [], "decision": "REJECTED", "reason": "novelty"}]})
    assert [f.name for f in feats] == ["onto_x", "onto_y"] and feats[0].selected and not feats[1].selected
    assert feats[0].source == "a, b" and feats[0].stability.value == 0.8
    assert feats[1].stability.value is None and feats[1].stability.reason == Reason.NOT_REPORTED
    splits = semantic_splits_from_audit([{"node_id": 0, "selected_feature": ["f0", "f1"], "base_score": 0.17,
                                          "semantic_bonus": 0.028, "final_score": 0.2, "semantic_reason": "mesma família",
                                          "decision_changed": True}])
    assert splits[0].feature == "f0, f1" and splits[0].semantic_bonus.value == 0.028 and splits[0].decision_changed
    assert semantic_splits_from_audit(None) == []
