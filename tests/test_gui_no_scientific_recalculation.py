"""A camada de interface nunca recalcula resultados científicos (Parte 40).

Verificação estática dos módulos novos da GUI: sem sklearn/scipy, sem funções de métricas,
sem chamadas a ``predict``/``fit``; e um teste dinâmico: construir/apresentar/exportar um resultado
não altera o ``comparison_results`` do backend nem chama o modelo.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

GUI = Path(__file__).resolve().parents[1] / "gui"
MODULES = ("result_presenter.py", "result_builder.py", "audit_controller.py", "audit_panel.py", "export_results.py",
           "messages.py", "experiment_state.py", "strings.py")
FORBIDDEN_CALLS = {"predict", "predict_proba", "fit", "fit_transform", "score", "accuracy_score", "f1_score",
                   "balanced_accuracy_score", "precision_score", "recall_score", "confusion_matrix", "train_test_split"}


@pytest.mark.parametrize("name", MODULES)
def test_module_has_no_scientific_computation(name):
    tree = ast.parse((GUI / name).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert not any(a.name.split(".")[0] in {"sklearn", "scipy"} for a in node.names), name
        if isinstance(node, ast.ImportFrom):
            assert (node.module or "").split(".")[0] not in {"sklearn", "scipy"}, name
        if isinstance(node, ast.Call):
            fn = node.func
            called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
            assert called not in FORBIDDEN_CALLS, (name, called)


def test_building_presenting_and_exporting_never_touch_models(tmp_path):
    from gui import result_presenter as rp
    from gui.export_results import export_results
    from gui.result_builder import build_experiment_result

    class Tripwire:
        def __getattr__(self, item):
            raise AssertionError(f"a interface chamou o modelo: {item}")

    comparison = {"precision": {"mlp": {"accuracy": .9}, "trepan_original": {"accuracy": .8}}, "fidelity": {"trepan_original": {"overall_fidelity": .9}}}
    snapshot = json.dumps(comparison, sort_keys=True)
    app = SimpleNamespace(
        metrics_comparator=SimpleNamespace(comparison_results=comparison), ontology_acceptance=None, ontology_quality_report=None,
        ontology_reasoner_report=None, loaded_ontology=None, loaded_ontology_path=None, mlp_model=Tripwire(),
        trepan_original_tree=Tripwire(), trepan_reloaded_tree=None, trepan_original_audit={"node_count": 5},
        trepan_reloaded_audit=None, onto_feature_bias_weight=0.5, ontology_match_threshold=0.7, dataset_fingerprint="x",
        current_seed=42, dataset_info={"name": "d"})
    result = build_experiment_result(app)
    rp.render_text(result, rp.SCIENTIFIC)
    rp.metrics_tables(result)
    export_results(result, str(tmp_path))
    assert json.dumps(comparison, sort_keys=True) == snapshot
    assert result.models["mlp_original"].metrics["accuracy"].value == .9
