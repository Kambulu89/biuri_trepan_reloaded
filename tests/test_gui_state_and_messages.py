"""Máquina de estados, resultados desatualizados e mensagens (Partes 3, 18, 21, 22, 24, 39)."""
import logging

import pytest

from core.experiment_result import (EnrichmentInfo, ExperimentResult, ExperimentState as S, Measure, ModelCard,
                                    Provenance, TreeDiagnostics)
from gui.experiment_state import ExperimentStateMachine, Fingerprint, InvalidTransition, StaleTracker
from gui.messages import Level, MessageLog, derive_messages, format_error


def test_initial_state_disables_everything_but_load():
    sm = ExperimentStateMachine()
    acts = sm.available_actions()
    assert acts["load_data"][0] is True
    for name, (ok, why) in acts.items():
        if name != "load_data":
            assert not ok and why, name


def test_buttons_follow_real_state():
    sm = ExperimentStateMachine()
    sm.transition(S.DATA_LOADED)
    assert sm.action_enabled("train")[0] and not sm.action_enabled("explain")[0]
    sm.transition(S.MODEL_TRAINED)
    assert sm.action_enabled("explain")[0] and not sm.action_enabled("visualize")[0]
    sm.transition(S.TREES_BUILT)
    assert sm.action_enabled("visualize")[0] and sm.action_enabled("compare")[0]
    # a GUI pode confirmar que as árvores realmente não existem
    assert not sm.action_enabled("visualize", has_trees=False)[0]
    sm.transition(S.RESULTS_READY)
    assert sm.action_enabled("export_results")[0]


def test_invalid_transition_rejected():
    sm = ExperimentStateMachine()
    with pytest.raises(InvalidTransition):
        sm.transition(S.TREES_BUILT)
    assert sm.state == S.NO_DATA


def test_busy_disables_actions_and_label():
    sm = ExperimentStateMachine()
    sm.transition(S.DATA_LOADED)
    sm.begin()
    ok, why = sm.action_enabled("train")
    assert not ok and why
    assert "A executar" in sm.label()
    sm.end()
    assert sm.action_enabled("train")[0]


def test_error_and_recover_preserve_last_good_state():
    sm = ExperimentStateMachine()
    sm.transition(S.DATA_LOADED)
    sm.transition(S.MODEL_TRAINED)
    sm.begin()
    sm.fail("boom")
    assert sm.state == S.ERROR and not sm.busy and sm.error_message == "boom"
    assert sm.reached(S.MODEL_TRAINED) and not sm.reached(S.TREES_BUILT)
    assert sm.recover() == S.MODEL_TRAINED


def test_retrain_after_results_goes_back():
    sm = ExperimentStateMachine()
    for s in (S.DATA_LOADED, S.MODEL_TRAINED, S.TREES_BUILT, S.RESULTS_READY):
        sm.transition(s)
    sm.transition(S.DATA_LOADED)  # novo dataset
    assert not sm.action_enabled("export_results")[0]


def test_stale_tracker_reasons_for_each_change():
    t = StaleTracker()
    base = Fingerprint("d1", "o1", "c1", 42)
    assert not t.is_stale(base)  # sem resultado -> nada a invalidar
    t.mark_result(base)
    assert not t.is_stale(base)
    assert t.reasons(Fingerprint("d2", "o1", "c1", 42)) == ["stale.dataset_changed"]
    assert t.reasons(Fingerprint("d1", "o2", "c1", 42)) == ["stale.owl_changed"]
    assert t.reasons(Fingerprint("d1", "o1", "c2", 42)) == ["stale.config_changed"]
    assert t.reasons(Fingerprint("d1", "o1", "c1", 7)) == ["stale.seed_changed"]
    banner = t.banner(Fingerprint("d2", "o2", "c1", 42))
    assert "DESATUALIZADOS" in banner and "dataset" in banner and "ontologia" in banner
    t.clear()
    assert t.banner(Fingerprint("x")) == ""


def _result(**kw):
    r = ExperimentResult(provenance=Provenance(experiment_id="abc123", seed=42))
    for k, v in kw.items():
        setattr(r, k, v)
    return r


def _tree(key, nodes):
    return TreeDiagnostics(tree=key, logical_nodes=Measure.of(nodes), available=True)


def test_small_sample_is_scientific_warning_not_error():
    r = _result(models={"c45": ModelCard(key="c45", evaluation_samples=12)})
    msgs = derive_messages(r)
    small = [m for m in msgs if m.code == "small_sample"]
    assert small and small[0].level == Level.SCIENTIFIC_WARNING.value
    assert all(m.level != Level.ERROR.value for m in msgs)
    assert small[0].experiment_id == "abc123"


def test_stump_is_info_diagnostic_never_tree_error():
    r = _result(trees={"trepan_original": _tree("trepan_original", 1), "trepan_reloaded": _tree("trepan_reloaded", 9)})
    msgs = [m for m in derive_messages(r) if m.code == "small_tree"]
    assert len(msgs) == 1 and msgs[0].level == Level.INFO.value
    assert "TREPAN Original" in msgs[0].text and "erro" not in msgs[0].text.lower()


def test_rejected_enrichment_is_info():
    r = _result(enrichment=EnrichmentInfo(mlp_status="REJECTED", decision="REJECT_X"))
    m = [m for m in derive_messages(r) if m.code == "enrichment_rejected"]
    assert m and m[0].level == Level.INFO.value and "REJECT_X" in m[0].text


def test_format_error_structure_and_traceback_logged(caplog):
    try:
        raise ValueError("coluna alvo inválida")
    except ValueError as exc:
        with caplog.at_level(logging.ERROR, logger="biuri.gui"):
            msg = format_error(exc, what_key="error.training", where="Treino do MLP",
                               action_key="error.action.training", experiment_id="abc123")
    assert msg.level == "ERROR" and msg.experiment_id == "abc123" and msg.where == "Treino do MLP"
    assert "ValueError: coluna alvo inválida" in msg.details and "Traceback" in msg.details
    assert msg.action and msg.text
    assert any("Traceback" in r.getMessage() and "abc123" in r.getMessage() for r in caplog.records)


def test_format_error_from_string_traceback():
    msg = format_error("Traceback (most recent call last):\n  File x\nKeyError: 'y'", experiment_id="e1")
    assert "KeyError" in msg.details


def test_message_log_filters_and_capacity():
    log = MessageLog(capacity=3)
    for i in range(5):
        log.add(format_error(f"e{i}", experiment_id="A" if i % 2 else "B"))
    assert len(log.items()) == 3
    assert all(m.experiment_id == "A" for m in log.items(experiment_id="A"))
    lines = log.format_lines()
    assert len(lines) == 3 and all("[ERRO]" in line for line in lines)


def test_budget_exhausted_stump_gets_a_scientific_warning_naming_the_cause():
    diag = TreeDiagnostics(tree="trepan_original", logical_nodes=Measure.of(3), queries_used=Measure.of(2000),
                           query_budget=Measure.of(2000), query_budget_exhausted=True,
                           stop_reasons={"STOP_QUERY_BUDGET_EXHAUSTED": 1, "STOP_PURE_NODE": 1}, available=True)
    msgs = derive_messages(_result(trees={"trepan_original": diag}))
    w = [m for m in msgs if m.code == "budget_limited"]
    assert w and w[0].level == Level.SCIENTIFIC_WARNING.value
    assert "2000/2000" in w[0].text and "TREPAN Original" in w[0].text and "3 nós" in w[0].text
    ok = TreeDiagnostics(tree="trepan_original", logical_nodes=Measure.of(15), query_budget_exhausted=False,
                         stop_reasons={"STOP_PURE_NODE": 4}, available=True)
    assert not [m for m in derive_messages(_result(trees={"trepan_original": ok})) if m.code == "budget_limited"]


def test_surrogate_accuracy_above_mlp_is_flagged_not_hidden_or_changed():
    mlp = ModelCard(key="mlp_original", status="AVAILABLE", evaluation_samples=45, metrics={"accuracy": Measure.of(0.93)})
    tree = ModelCard(key="trepan_original", status="AVAILABLE", evaluation_samples=45, metrics={"accuracy": Measure.of(0.944)})
    r = _result(models={"mlp_original": mlp, "trepan_original": tree})
    w = [m for m in derive_messages(r) if m.code == "surrogate_above_oracle"]
    assert w and w[0].level == Level.SCIENTIFIC_WARNING.value
    assert "94.4%" in w[0].text and "93.0%" in w[0].text and "TREPAN Original" in w[0].text and "45" in w[0].text
    assert tree.metrics["accuracy"].value == 0.944            # o valor medido nunca é alterado
    tree2 = ModelCard(key="trepan_original", status="AVAILABLE", metrics={"accuracy": Measure.of(0.90)})
    assert not [m for m in derive_messages(_result(models={"mlp_original": mlp, "trepan_original": tree2}))
                if m.code == "surrogate_above_oracle"]


def test_unstable_or_failed_structure_tuning_is_flagged():
    unstable = _result(config={"trepan_tuning": {"selected": {"purity_epsilon": 0.02, "max_nodes": 31}, "stable": False,
                                                 "selection_probability": 0.33, "threshold": 0.6}})
    w = [m for m in derive_messages(unstable) if m.code == "tuning_uncertain"]
    assert w and w[0].level == Level.SCIENTIFIC_WARNING.value and "33%" in w[0].text and "60%" in w[0].text
    failed = _result(config={"trepan_tuning": {"failed": "RuntimeError: boom", "fallback": "canonical_defaults"}})
    f = [m for m in derive_messages(failed) if m.code == "tuning_failed"]
    assert f and f[0].level == Level.WARNING.value and "canónica" in f[0].text
    ok = _result(config={"trepan_tuning": {"selected": {"purity_epsilon": 0.05, "max_nodes": 31}, "stable": True, "selection_probability": 1.0}})
    assert not [m for m in derive_messages(ok) if m.code in {"tuning_uncertain", "tuning_failed"}]
