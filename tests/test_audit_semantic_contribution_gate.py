"""Regressão (auditoria): o gate de contribuição semântica não pode declarar contribuição quando a ontologia não
alterou nenhuma decisão, e tem de declarar o âmbito da evidência (uma partição, sem controlos)."""
from core.semantic_contribution_gate import SemanticContributionGate

QUALITY = {"accepted": True, "status": "VALID_DOMAIN_ONTOLOGY"}


def _run(impact, efsr_accepted, dfid=0.007):
    return SemanticContributionGate().evaluate(
        ontology_quality=QUALITY,
        semantic_audit={"ontology_usage_rate": 1.0, "semantic_decision_impact": impact,
                        "error_focused_interventions_attempted": 15, "error_focused_interventions_accepted": efsr_accepted,
                        "error_focused_interventions_rejected": 15 - efsr_accepted},
        comparison={"delta_oracle_fidelity": dfid, "delta_balanced_accuracy": 0.009, "delta_macro_f1": 0.008},
        original_metrics={"nodes": 27}, reloaded_metrics={"nodes": 23})


def test_gain_without_any_semantic_decision_is_not_a_semantic_contribution():
    # Caso real observado (Breast Cancer, semente 7): fidelidade +0,007 com impacto 0 e 0 intervenções aceites.
    report = _run(impact=0.0, efsr_accepted=0)
    assert not report["accepted"]
    assert report["status"] == "NO_SEMANTIC_DECISION_CHANGED"
    assert "no_semantic_decision_changed" in report["reasons"]


def test_observed_effect_is_reported_with_its_evidence_scope():
    for report in (_run(impact=0.2, efsr_accepted=3), _run(impact=0.0, efsr_accepted=2)):
        assert report["accepted"] and report["status"] == "SEMANTIC_CONTRIBUTION_SUPPORTED"
        assert report["evidence_level"] == "observed_single_split"
        assert report["controls_evaluated"] is False and report["attributable_to_ontology"] is False


def test_fidelity_loss_keeps_its_own_status():
    report = _run(impact=0.2, efsr_accepted=3, dfid=-0.01)
    assert not report["accepted"] and report["status"] == "SEMANTIC_EFFECT_WITHOUT_FIDELITY_GAIN"
