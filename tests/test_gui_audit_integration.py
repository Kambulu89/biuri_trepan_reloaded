"""Integração da camada de auditoria no BiuriApp real (offscreen, sem treino pesado)."""
import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication  # noqa: E402

from core.experiment_result import ExperimentState as S  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture()
def arff(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / "small.arff"
    lines = ["@relation small", "@attribute a numeric", "@attribute b numeric", "@attribute class {neg,pos}", "@data"]
    for _ in range(60):
        a, b = rng.normal(size=2)
        lines.append(f"{a:.3f},{b:.3f},{'pos' if a + b > 0 else 'neg'}")
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture()
def window(qapp):
    from gui.biuri_app_complete import BiuriApp
    w = BiuriApp()
    yield w
    w.close()


def test_audit_tab_exists_and_initial_state_has_no_data(window):
    tabs = [window.content_tabs.tabText(i) for i in range(window.content_tabs.count())]
    assert any("Auditoria" in t for t in tabs)
    assert window.audit.sm.state == S.NO_DATA
    assert window.btn_load_data.isEnabled()
    for btn in (window.btn_train_model, window.btn_generate_explanation, window.btn_visualize_tree,
                window.btn_compare_metrics, window.btn_export_results, window.btn_export_tree):
        assert not btn.isEnabled() and btn.toolTip()


def test_loading_data_updates_state_buttons_and_dataset_info(window, arff):
    window.load_data_from_file(arff)
    assert window.audit.sm.state == S.DATA_LOADED
    assert window.btn_train_model.isEnabled() and not window.btn_visualize_tree.isEnabled()
    info = window.dataset_info
    assert info["rows"] == 60 and info["features"] == 2 and info["classes"] == 2 and info["name"] == "small.arff"
    assert window.dataset_fingerprint and len(window.dataset_fingerprint) == 16


def test_export_buttons_are_separate_actions(window):
    assert window.btn_export_results is not window.btn_export_tree
    assert "dados" in window.btn_export_results.text() and "imagem" in window.btn_export_tree.text()


def test_bias_weight_change_marks_trained_result_stale(window, arff):
    window.load_data_from_file(arff)
    window.audit.on_training_started()
    window.mlp_model = object()
    window.trepan_original_tree = window.trepan_reloaded_tree = object()
    window.trepan_original_audit = {"node_count": 5}
    window.trepan_reloaded_audit = {"node_count": 5}
    window.audit.on_training_finished()
    window.audit.on_metrics_finished()
    assert window.audit.sm.state == S.RESULTS_READY and not window.audit.result.stale
    window._apply_onto_bias_weight(0.9)
    assert window.audit.result.stale and "stale.config_changed" in window.audit.result.stale_reasons
    assert "DESATUALIZADOS" in window.audit_panel.stale_banner.text()
    window.load_data_from_file(arff)  # novo dataset: o resultado anterior continua marcado como stale
    assert window.audit.result.stale and window.audit.sm.state == S.DATA_LOADED


def test_structured_failure_shows_id_and_keeps_buttons_usable(window, arff, monkeypatch):
    shown = []
    monkeypatch.setattr(type(window), "_show_structured_error", lambda self, msg: shown.append(msg))
    window.load_data_from_file(arff)
    eid = window.audit.on_training_started()
    window._pending_failure_detail = "Traceback (most recent call last):\nValueError: boom"

    class Dlg:
        def close(self):
            pass

    window._on_training_failed("boom", Dlg())
    assert shown and shown[0].experiment_id == eid and "ValueError" in shown[0].details
    assert eid in window.audit_panel.error_banner.text()
    assert window.btn_train_model.isEnabled() and window.audit.sm.state == S.DATA_LOADED


def test_training_worker_emits_failure_detail_before_failed(qapp):
    from gui.training_worker import TrainingWorker
    order = []

    class Boom:
        def _execute_training_pipeline(self, *a, **k):
            raise RuntimeError("falhou")

    from core.training_config import enforce_scientific_preset
    w = TrainingWorker(Boom(), _preset())
    w.failed_detail.connect(lambda m, tb: order.append(("detail", m, "RuntimeError" in tb)))
    w.failed.connect(lambda m: order.append(("failed", m)))
    w.run()
    assert order == [("detail", "falhou", True), ("failed", "falhou")]


def _preset():
    from core.training_config import enforce_scientific_preset
    return enforce_scientific_preset(use_cache=False)


def test_metrics_widget_never_shows_zero_fidelity_for_models_without_oracle(qapp):
    from gui.biuri_app_complete import MetricsComparisonWidget
    w = MetricsComparisonWidget()
    block = {"accuracy": .9, "precision_macro": .9, "f1_macro": .9, "balanced_accuracy": .9}
    w.update_comparison_results({
        "precision": {"mlp": block, "trepan_original": block, "c45_j48": block},
        "fidelity": {"trepan_original": {"overall_fidelity": .95, "fidelity_reference": "mlp_original"},
                     "c45_j48": {"overall_fidelity": .8, "fidelity_reference": "mlp_original"}},
        "model_info": {"n_test_samples": 40}})
    data = w.metrics_visualizer.models_data
    assert data["MLP Original"]["fidelity_kind"] is None
    assert data["Trepan-Original"]["fidelity_kind"] == "oracle"
    assert data["C4.5-Nativo"]["fidelity_kind"] == "agreement"
    from gui.result_presenter import fidelity_label_text
    assert fidelity_label_text(None, None).startswith("Fidelity: Não aplicável")
    assert fidelity_label_text("agreement", 80.0).startswith("Concordância com o MLP (diagnóstico)")
    assert fidelity_label_text("oracle", 95.0) == "Fidelity: 95.0%"
