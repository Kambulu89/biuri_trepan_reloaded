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


class _Dlg:
    def close(self):
        pass


def _finish(w, result=None):
    w._refresh_dataset_split_info = lambda: None
    w._unlock_clarity = lambda *a, **k: None
    w._refresh_counterfactual_panel = lambda: None
    w._on_training_finished(result or {"success": True}, _Dlg())


def test_ontology_change_discards_all_dependent_artifacts(app_with_models):
    w, _ = app_with_models
    w.c45_tree = object()
    w.mlp_model_onto = object()
    w.X_encoded_aug = [[0.0]]
    w.benchmark_outcome = object()
    w.cf_result = {"x": 1}
    w.metrics_comparator.comparison_results = {"k": 1}
    w._ontology_changed("teste")
    for name in ("mlp_model", "mlp_model_onto", "trepan_original_tree", "trepan_reloaded_tree", "c45_tree",
                 "X_encoded", "y_encoded", "X_encoded_aug", "selected_oracle", "benchmark_outcome", "cf_result"):
        assert getattr(w, name) is None, name
    assert w.metrics_comparator.comparison_results == {}
    assert not w._has_any_trained_model()


def test_ontology_removal_invalidates_too(app_with_models):
    w, _ = app_with_models
    w._clear_loaded_ontology()
    assert w._stale_models_reason
    assert w.mlp_model is None and w.trepan_reloaded_tree is None


def test_successful_training_for_current_state_unlocks(app_with_models):
    w, _ = app_with_models
    w._ontology_changed("teste")
    w._training_started_for_current_state()
    w.mlp_model = object()                       # modelo reconstruído pelo treino
    _finish(w)
    assert w._stale_models_reason is None
    assert w.trepan_reloaded_tree is None        # árvores antigas NÃO regressam: só o que foi reconstruído existe
    assert not w._has_trained_models_for_cf()


def test_partial_training_does_not_validate_old_artifacts(app_with_models):
    w, warnings = app_with_models
    w._ontology_changed("teste")
    w._training_started_for_current_state()
    _finish(w)                                   # sucesso declarado, mas nenhum modelo reconstruído
    assert w._stale_models_reason
    assert w.mlp_model is None and warnings


def test_ontology_change_during_training_invalidates_result(app_with_models):
    w, warnings = app_with_models
    w._training_started_for_current_state()
    w.mlp_model = object()
    w._ontology_changed("durante o treino")
    w.mlp_model = object()                       # o worker acabou de escrever um modelo treinado sob a ontologia antiga
    _finish(w)
    assert w._stale_models_reason and w.mlp_model is None and warnings


def test_no_ontology_mode_still_operational(qapp):
    w = BiuriApp()
    try:
        assert w._stale_models_reason is None
        w._training_started_for_current_state()
        w.mlp_model = object()
        w.X_encoded, w.y_encoded = [[0.0]], [0]
        _finish(w)
        assert w._stale_models_reason is None and w.mlp_model is not None
        assert w._require_current_models("x") is True
    finally:
        w.close()


def test_benchmark_finished_discards_result_if_ontology_changed_meanwhile(app_with_models):
    w, warnings = app_with_models
    w._training_started_for_current_state()
    w._ontology_changed("durante o benchmark")

    class Outcome:
        def tree_view(self):
            raise AssertionError("resultado de um estado ontológico antigo não pode ser instalado")
    w._on_benchmark_finished({"outcome": Outcome()}, _Dlg())
    assert warnings and w.benchmark_outcome is None and w.trepan_reloaded_tree is None
