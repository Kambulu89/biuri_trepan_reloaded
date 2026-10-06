"""SCIENTIFIC / BENCHMARK na GUI: «Comparar métricas» apresenta as métricas do pipeline científico (sem recalcular com o estado interativo)
e a visualização inclui o C4.5 avaliado. Regressão: «OrderedLabelEncoder ainda não foi ajustado» ao comparar métricas após o treino científico."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, str(Path(__file__).parent))

import core.scientific_benchmark_service as svc  # noqa: E402
from gui.benchmark_metrics import comparison_results_from_report  # noqa: E402
from test_scientific_benchmark_mode import _fast_search  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt6")
    from PyQt6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(scope="module")
def arff3(tmp_path_factory):
    rng = np.random.default_rng(1)
    path = tmp_path_factory.mktemp("g") / "tres_classes.arff"
    lines = ["@relation t3", "@attribute a numeric", "@attribute b numeric", "@attribute c numeric", "@attribute class {neg,mid,pos}", "@data"]
    for _ in range(150):
        a, b, c = rng.normal(size=3)
        lines.append(f"{a:.3f},{b:.3f},{c:.3f},{'pos' if a + 0.5 * b > 0.5 else 'mid' if a + 0.5 * b > -0.5 else 'neg'}")
    path.write_text("\n".join(lines))
    return str(path)


@pytest.fixture(scope="module")
def trained(qapp, arff3):
    from gui.biuri_app_complete import BiuriApp
    real = svc.run_scientific_benchmark
    svc.run_scientific_benchmark = lambda df, **kw: real(df, **{**kw, "search": _fast_search()})
    try:
        w = BiuriApp()
        w.load_data_from_file(arff3)
        w.execution_mode_combo.setCurrentIndex(w.execution_mode_combo.findData("SCIENTIFIC_BENCHMARK"))
        outcome = w._execute_benchmark_pipeline()

        class _Dlg:
            def close(self):
                pass
        w._on_benchmark_finished({"outcome": outcome}, _Dlg())
        yield w, outcome
    finally:
        svc.run_scientific_benchmark = real
        w.close()


def test_production_pipeline_evaluates_the_c45_against_the_same_frozen_oracle(trained):
    _, outcome = trained
    c45 = outcome.report["evaluation"]["models"]["c45_native"]
    assert 0.0 <= c45["oracle_fidelity"] <= 1.0 and abs(c45["oracle_fidelity"] + c45["disagreement_rate_to_oracle"] - 1.0) < 1e-12
    assert c45["fidelity_oracle_id"] == outcome.oracle_id and c45["leaves"] >= 1 and c45["depth"] >= 0
    card = outcome.result.models["c45"]
    assert card.agreement_with_mlp.value == pytest.approx(c45["oracle_fidelity"]) and card.fidelity.value is None   # sem inventar «fidelity» de treino
    assert outcome.artifacts["c45_native"] is outcome.tree_view()["c45_tree"]                                    # o objeto EXATO avaliado


def test_compare_metrics_in_benchmark_mode_shows_the_pipeline_metrics_without_recomputing(trained, monkeypatch):
    w, outcome = trained
    monkeypatch.setattr(type(w), "_execute_metrics_comparison", lambda *a, **k: (_ for _ in ()).throw(AssertionError("recalculou na interface")))
    w.compare_metrics()
    assert w.content_tabs.currentWidget() is w.metrics_tab
    assert outcome.oracle_id[:8] in w.statusBar().currentMessage() and "sem recálculo" in w.statusBar().currentMessage()
    models = {m for m in w.metrics_widget.metrics_visualizer.model_data} if hasattr(w.metrics_widget.metrics_visualizer, "model_data") else None
    if models is not None:
        assert {"MLP Original", "Trepan-Original", "Trepan-Reloaded", "C4.5-Nativo"} <= models


def test_visualization_offers_the_c45_tree(trained):
    w, outcome = trained
    w.visualize_tree()
    options = dict(w.tree_widget.tree_options)
    assert {"Trepan-Original", "Trepan-Reloaded", "C4.5-Nativo"} <= set(options)
    assert options["C4.5-Nativo"] is outcome.artifacts["c45_native"]
    assert w.c45_tree is outcome.artifacts["c45_native"]


def test_the_interactive_mode_does_not_inherit_the_benchmark_c45(trained):
    w, _ = trained
    w._clear_benchmark_view()
    assert w.benchmark_view is None and w.c45_tree is None


def test_converter_only_reformats_numbers_already_computed():
    report = {"manifest": {"test_rows": 40}, "evaluation": {"models": {
        "mlp_original": {"accuracy": .9, "balanced_accuracy": .9, "macro_f1": .89, "precision_macro": .91, "recall_macro": .9},
        "original": {"accuracy": .8, "balanced_accuracy": .8, "macro_f1": .79, "precision_macro": .8, "recall_macro": .8, "oracle_fidelity": .85},
        "c45_native": {"accuracy": .7, "balanced_accuracy": .7, "macro_f1": .69, "precision_macro": .7, "recall_macro": .7, "oracle_fidelity": .75}}}}
    out = comparison_results_from_report(report, "abcdef0123456789")
    assert out["precision"]["c45_j48"]["f1_macro"] == .69 and out["fidelity"]["c45_j48"]["overall_fidelity"] == .75
    assert "mlp" not in out["fidelity"] and "trepan_reloaded" not in out["precision"] and out["model_info"]["n_test_samples"] == 40
    assert "abcdef01" in out["fidelity"]["trepan_original"]["fidelity_reference"]
