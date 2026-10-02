"""Estados independentes (estrutura, mapeamento, novidade, MLP, TREPAN) e painel de diagnóstico."""
import pytest

from core.ontology_stage_status import build_ontology_stage_status
from gui.ontology_status_presenter import build_ontology_status_text, build_semantic_diagnostics_text


def _quality(accepted=True, abox_ok=True, mapped=30, total=30, status="VALID_DOMAIN_ONTOLOGY", ambiguous=0):
    return {
        "accepted": accepted, "status": status,
        "abox": {"accepted": abox_ok, "status": "SAFE" if abox_ok else "REJECT_TEST_INSTANCE_LEAKAGE"},
        "metrics": {"mapped_features": mapped, "total_features": total, "feature_coverage": mapped / total,
                    "ambiguous_matches": ambiguous, "entity_collisions": 0, "generic_match_ratio": 0.0,
                    "knowledge_split": {"tbox": {"classes": 4, "datatype_properties": 30, "object_properties": 0},
                                        "abox": {"individuals": 0, "status": "SAFE"}},
                    "semantic_richness": {"level": "limited", "knowledge_sources": ["statistic_roles"]}},
        "issues": [],
    }


def _report(decision="REJECT_NO_INFORMATIONAL_GAIN", accepted=False):
    row = {"accuracy": .9, "balanced_accuracy": .9, "macro_f1": .9, "recall_macro": .9,
           "minority_recall": .85, "precision_macro": .9, "utility": .85}
    return {
        "decision": decision, "decision_reason": "ci_includes_zero_and_no_secondary_gain",
        "semantic_mlp_accepted": accepted, "semantic_trepan_available": True,
        "selected_semantic_features": ["onto_a"] if accepted else [],
        "ontology_knowledge_available_for_trepan": True,
        "reasoner": {"reasoner_used": True, "consistent": True, "duration_seconds": 0.5, "inferred_axioms_count": 3},
        "stages": {
            "B_novelty": {"generated": 34, "removed_constant": 2, "removed_duplicate": 3, "retained": 29,
                          "retained_ontology_knowledge": 29, "retained_statistical": 0, "status": "VALID"},
            "C_screening": {"stable_features": ["onto_a", "onto_b"]},
            "D_mlp_comparison": {"base": row, "with_owl": {**row, "utility": .86},
                                 "delta": {k: 0.0 for k in row if k != "utility"}, "utility_gain": 0.01,
                                 "utility_gain_ci": [-0.01, 0.03], "confidence": 0.95},
        },
    }


def test_mlp_rejected_does_not_disable_trepan():
    st = build_ontology_stage_status(_quality(), {}, _report())
    assert st["ontology_structural_status"] == "VALID"
    assert st["mapping_status"] == "VALID" and st["abox_status"] == "SAFE"
    assert st["semantic_novelty_status"] == "VALID"
    assert st["mlp_enrichment_status"] == "REJECT_NO_INFORMATIONAL_GAIN"
    assert st["semantic_mlp_accepted"] is False
    assert st["semantic_trepan_available"] is True
    assert st["semantic_trepan_reason"] == "structure_valid_mapped_and_leak_free"


def test_mlp_accepted_and_trepan_available_are_independent_fields():
    st = build_ontology_stage_status(_quality(), {}, _report("ACCEPT_SIGNIFICANT_GAIN", True))
    assert st["semantic_mlp_accepted"] is True and st["semantic_trepan_available"] is True


@pytest.mark.parametrize("quality,reason", [
    (_quality(accepted=False, status="GENERIC_ONTOLOGY"), "ontology_quality:GENERIC_ONTOLOGY"),
    (_quality(accepted=False, abox_ok=False, status="CONTAMINATED_ONTOLOGY"), "ontology_quality:CONTAMINATED_ONTOLOGY"),
])
def test_invalid_ontology_disables_trepan_with_reason(quality, reason):
    st = build_ontology_stage_status(quality, {}, None)
    assert st["semantic_trepan_available"] is False and st["semantic_trepan_reason"] == reason


def test_abox_leak_blocks_trepan_even_if_other_gates_pass():
    q = _quality(abox_ok=False)  # aceite estruturalmente mas ABox com leakage
    st = build_ontology_stage_status(q, {}, None)
    assert st["semantic_trepan_available"] is False and st["semantic_trepan_reason"].startswith("abox:")


def test_ambiguity_is_visible_in_mapping_status_not_hidden_by_coverage():
    st = build_ontology_stage_status(_quality(ambiguous=2), {}, None)
    assert st["mapping_status"] == "VALID_WITH_AMBIGUITY"


def test_legacy_keys_and_calls_still_work():
    st = build_ontology_stage_status(_quality(), {"ontology_feature_gate_accepted": True})
    assert st["trepan_semantic_use_allowed"] is True and st["ontology_feature_engineering_accepted"] is True
    assert st["semantic_mlp_accepted"] is True  # sem relatório novo: usa o gate legado


def test_panel_explains_decision_with_numbers_not_just_the_verdict():
    text = build_semantic_diagnostics_text(_quality(), _report())
    for needle in ("ONTOLOGIA", "FEATURES SEMÂNTICAS", "VALIDAÇÃO SEMÂNTICA DO MLP", "SEMÂNTICA PARA O TREPAN",
                   "30 / 30", "ABox: SAFE", "CONSISTENTE", "3 axiomas inferidos", "Geradas: 34",
                   "constant=2", "duplicate=3", "Retidas (novidade): 29", "Estáveis: 2",
                   "Balanced Accuracy", "IC 95,0%", "Decisão: REJECT_NO_INFORMATIONAL_GAIN",
                   "Causa:", "Disponível: SIM", "independente da disponibilidade"):
        assert needle in text, needle


def test_panel_shows_reasoner_fallback_when_not_executed():
    rep = _report(); rep["reasoner"] = {"reasoner_used": False, "executed": False, "fallback": "explicit_axioms_only"}
    assert "NÃO EXECUTADO — fallback: explicit_axioms_only" in build_semantic_diagnostics_text(_quality(), rep)


def test_status_text_appends_panel_only_when_enrichment_is_given():
    base = build_ontology_status_text(_quality(), {})
    assert "VALIDAÇÃO SEMÂNTICA DO MLP" not in base  # saída antiga inalterada
    full = build_ontology_status_text(_quality(), {}, enrichment_report=_report())
    assert full.startswith(base) and "VALIDAÇÃO SEMÂNTICA DO MLP" in full
