"""Fluxo completo do controlador sem Qt: estados, botões, stale, cache, erros, mensagens."""
from types import SimpleNamespace

import pytest

from core.experiment_result import ExperimentState as S
from gui.audit_controller import AuditController, dataset_fingerprint_of, progress_text
from gui.audit_controller import BUTTON_ATTRS


class Btn:
    def __init__(self):
        self.enabled, self.tip = True, ""

    def setEnabled(self, flag):
        self.enabled = bool(flag)

    def setToolTip(self, text):
        self.tip = text


class Panel:
    def __init__(self):
        self.result, self.state, self.error, self.lines = None, "", "", []

    def set_result(self, r):
        self.result = r

    def set_state(self, t):
        self.state = t

    def set_error(self, t):
        self.error = t

    def set_log_lines(self, lines):
        self.lines = list(lines)


def make_app(**kw):
    app = SimpleNamespace(
        metrics_comparator=SimpleNamespace(comparison_results={}), ontology_acceptance=None, ontology_quality_report=None,
        ontology_reasoner_report=None, loaded_ontology=None, loaded_ontology_path=None, mlp_model=None,
        trepan_original_tree=None, trepan_reloaded_tree=None, trepan_original_audit=None, trepan_reloaded_audit=None,
        onto_feature_bias_weight=0.5, ontology_match_threshold=0.72, dataset_fingerprint="d1", current_seed=42,
        dataset_info={"name": "iris", "rows": 150, "features": 4, "classes": 3})
    for attr in BUTTON_ATTRS.values():
        setattr(app, attr, Btn())
    for k, v in kw.items():
        setattr(app, k, v)
    return app


def trained(app):
    app.mlp_model = object()
    app.trepan_original_tree = app.trepan_reloaded_tree = object()
    app.trepan_original_audit = {"node_count": 9, "depth": 3}
    app.trepan_reloaded_audit = {"node_count": 11, "depth": 3}


def test_buttons_reflect_state_through_lifecycle():
    app, panel = make_app(), Panel()
    c = AuditController(app, panel)
    c.apply_buttons()
    assert app.btn_load_data.enabled and not app.btn_train_model.enabled and "Carregue" in app.btn_train_model.tip
    c.on_data_loaded()
    assert app.btn_train_model.enabled and not app.btn_generate_explanation.enabled
    c.on_training_started()
    assert not app.btn_train_model.enabled and "executar" in panel.state.lower()
    trained(app)
    c.on_training_finished()
    assert app.btn_generate_explanation.enabled and app.btn_visualize_tree.enabled and app.btn_export_results.enabled
    assert c.sm.state == S.TREES_BUILT and panel.result.state == "TREES_BUILT"


def test_semantic_validated_state_when_acceptance_present():
    app = make_app()
    c = AuditController(app, Panel())
    c.on_data_loaded()
    c.on_training_started()
    trained(app)
    app.ontology_acceptance = {"accepted": False, "reason": "ONTOLOGY_VALID_BUT_NO_PREDICTIVE_UTILITY"}
    app.loaded_ontology = object()
    c.on_training_finished()
    assert S.SEMANTIC_VALIDATED in c.sm.history
    assert c.result.enrichment.mlp_status == "REJECTED"


def test_metrics_finished_reaches_results_ready():
    app = make_app()
    c = AuditController(app, Panel())
    c.on_data_loaded(); c.on_training_started(); trained(app); c.on_training_finished()
    app.metrics_comparator.comparison_results = {"precision": {"mlp": {"accuracy": .9}}, "fidelity": {}}
    r = c.on_metrics_finished()
    assert c.sm.state == S.RESULTS_READY and r.models["mlp_original"].metrics["accuracy"].value == .9
    assert r.provenance.experiment_id == c.experiment_id


def test_stale_when_owl_config_or_dataset_changes_until_retrain():
    app, panel = make_app(), Panel()
    c = AuditController(app, panel)
    c.on_data_loaded(); c.on_training_started(); trained(app); c.on_training_finished()
    assert c.check_stale() is False and not c.result.stale
    app.onto_feature_bias_weight = 0.9
    assert c.check_stale() is True and c.result.stale and "stale.config_changed" in c.result.stale_reasons
    assert panel.result is c.result
    app.dataset_fingerprint = "other"
    c.check_stale()
    assert set(c.result.stale_reasons) == {"stale.config_changed", "stale.dataset_changed"}
    assert any("desatualizados" in l.lower() for l in panel.lines)
    # novo treino limpa o stale
    c.on_training_started(); c.on_training_finished()
    assert c.result.stale is False and c.check_stale() is False


def test_new_dataset_marks_previous_result_stale_not_current():
    app = make_app()
    c = AuditController(app, Panel())
    c.on_data_loaded(); c.on_training_started(); trained(app); c.on_training_finished()
    app.dataset_fingerprint = "d2"
    app.mlp_model = app.trepan_original_tree = app.trepan_reloaded_tree = None
    c.on_data_loaded()
    assert c.result.stale and c.sm.state == S.DATA_LOADED
    assert not app.btn_visualize_tree.enabled and app.btn_train_model.enabled


def test_cache_indicator_flows_to_result_and_not_stale_if_config_changes_are_flagged():
    app = make_app(_last_cache_info={"used": True, "key": "abcdef0123456789"})
    c = AuditController(app, Panel())
    c.on_data_loaded(); c.on_training_started(); trained(app)
    r = c.on_training_finished()
    assert r.provenance.cache_used and r.provenance.cache_key == "abcdef0123456789"
    app.ontology_match_threshold = 0.5   # config mudou: o resultado em cache já não é "atual"
    assert c.check_stale() and c.result.stale


def test_failure_is_structured_and_unblocks_buttons():
    app, panel = make_app(), Panel()
    c = AuditController(app, panel)
    c.on_data_loaded()
    eid = c.on_training_started()
    msg = c.on_failure("Traceback (most recent call last):\nValueError: x", what_key="error.training",
                       where_key="error.training_where", action_key="error.action.training")
    assert msg.experiment_id == eid and msg.where == "Treino do pipeline" and "ValueError" in msg.details
    assert eid in panel.error and "O treino do modelo falhou" in panel.error
    assert c.sm.state == S.DATA_LOADED and not c.sm.busy and app.btn_train_model.enabled
    assert any(eid in l for l in panel.lines)


def test_stump_reports_diagnostic_message_not_error():
    app = make_app()
    c = AuditController(app, Panel())
    c.on_data_loaded(); c.on_training_started(); trained(app)
    app.trepan_original_audit = {"node_count": 1, "depth": 0, "stop_summary": {"loop_end_reason": "no_expandable_nodes_left"}}
    r = c.on_training_finished()
    small = [m for m in r.messages if m.code == "small_tree"]
    assert small and small[0].level == "INFO"
    assert not any(m.level == "ERROR" for m in r.messages)


def test_progress_text_by_stage_without_percentages():
    assert progress_text("mlp") == "A treinar o MLP…"
    assert progress_text("trepan_original") == "A construir o TREPAN Original…"
    assert progress_text("trepan_reloaded") == "A construir o TREPAN Reloaded…"
    assert progress_text("compare") == "A calcular métricas…"
    assert progress_text("???") == "A trabalhar…" and "%" not in progress_text("mlp")


def test_dataset_fingerprint_is_stable_and_sensitive():
    import numpy as np
    X = np.array([[1, "a"], [2, "b"]], dtype=object)
    y = np.array(["x", "y"], dtype=object)
    assert dataset_fingerprint_of(X, y) == dataset_fingerprint_of(X.copy(), y.copy())
    X2 = X.copy(); X2[0, 0] = 3
    assert dataset_fingerprint_of(X2, y) != dataset_fingerprint_of(X, y)
