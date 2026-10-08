"""Modelos treinados sob um estado ontológico não podem ser reutilizados depois de a ontologia mudar (árvores, métricas, contrafactuais, exportações)."""
import os

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import pytest

pytest.importorskip('PyQt6')

from PyQt6.QtWidgets import QApplication, QMessageBox

from gui.biuri_app_complete import BiuriApp

GUARDED = [
    "generate_explanation", "visualize_tree", "compare_metrics", "show_natural_explanations",
    "export_results", "export_tree", "improve_surrogate_action", "generate_counterfactuals_action",
]


@pytest.fixture(scope='module')
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def app_with_models(qapp, monkeypatch):
    warnings = []
    monkeypatch.setattr(QMessageBox, "warning", lambda *a, **k: warnings.append(a[2:]) or QMessageBox.StandardButton.Ok)
    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: QMessageBox.StandardButton.Ok)
    w = BiuriApp()
    w.mlp_model = object()
    w.trepan_original_tree = object()
    w.trepan_reloaded_tree = object()
    w.X_encoded = [[0.0]]
    w.y_encoded = [0]
    yield w, warnings
    w.close()


def test_fresh_models_are_not_stale(app_with_models):
    w, _ = app_with_models
    assert w._stale_models_reason is None
    assert w._has_trained_models_for_cf()
    assert w._require_current_models("x") is True


def test_ontology_change_marks_models_stale_and_blocks_cf(app_with_models):
    w, _ = app_with_models
    w._ontology_changed("teste")
    assert w._stale_models_reason
    assert not w._has_trained_models_for_cf()
    with pytest.raises(ValueError):
        w._build_cf_session()


@pytest.mark.parametrize("handler", GUARDED)
def test_every_downstream_action_is_blocked_when_stale(app_with_models, handler):
    w, warnings = app_with_models
    w._ontology_changed("teste")
    getattr(w, handler)()
    assert warnings, handler
    assert any("ontologia mudou" in str(x) for x in warnings), handler


def test_ontology_change_without_models_does_not_flag(qapp):
    w = BiuriApp()
    try:
        w._ontology_changed("teste")
        assert w._stale_models_reason is None
    finally:
        w.close()


def test_successful_training_clears_stale_flag(app_with_models):
    w, _ = app_with_models
    w._ontology_changed("teste")

    class Dlg:
        def close(self):
            pass
    w._refresh_dataset_split_info = lambda: None
    w._unlock_clarity = lambda *a, **k: None
    w._refresh_counterfactual_panel = lambda: None
    w._on_training_finished({"success": True}, Dlg())
    assert w._stale_models_reason is None
