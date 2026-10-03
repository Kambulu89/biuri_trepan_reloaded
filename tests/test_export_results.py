"""Exportação de resultados: ficheiros estruturados, stale e separação da exportação da árvore."""
import csv
import json
from pathlib import Path

from gui.export_results import FILES, export_results, metrics_rows
from tests.test_result_presenter import make_result


def test_export_writes_all_structured_files(tmp_path):
    r = make_result()
    r.counterfactual = {"n": 3}
    out = export_results(r, str(tmp_path / "x"))
    for name in FILES + ("counterfactual_summary.json",):
        assert Path(out[name]).exists(), name
    manifest = json.loads(Path(out["manifest.json"]).read_text())
    assert manifest["experiment_id"] == r.provenance.experiment_id and manifest["stale"] is False
    assert json.loads(Path(out["results.json"]).read_text())["provenance"]["seed"] == 42
    # nenhuma imagem de árvore: é outra ação
    assert not [p for p in Path(out["report.md"]).parent.iterdir() if p.suffix in (".png", ".svg", ".pdf")]


def test_metrics_csv_keeps_reason_for_missing_values_and_oracle(tmp_path):
    out = export_results(make_result(), str(tmp_path))
    rows = list(csv.DictReader(open(out["metrics.csv"], encoding="utf-8")))
    c45 = [x for x in rows if x["model"] == "c45" and x["metric"] == "fidelity"][0]
    assert c45["value"] == "" and c45["reason"] == "no_oracle" and c45["oracle"] == ""
    fid = [x for x in rows if x["model"] == "trepan_reloaded" and x["metric"] == "fidelity"][0]
    assert fid["oracle"] == "mlp_original" and float(fid["value"]) == 0.96
    assert {x["group"] for x in rows} >= {"predictive", "fidelity", "complexity", "diagnostic"}


def test_metrics_values_are_exactly_the_result_values():
    r = make_result()
    for model, group, metric, value, reason, oracle in metrics_rows(r):
        if group == "predictive" and value != "":
            assert value == r.models[model].metrics[metric].value


def test_stale_flag_and_warning_in_export(tmp_path):
    r = make_result()
    r.stale, r.stale_reasons = True, ["stale.dataset_changed"]
    out = export_results(r, str(tmp_path))
    assert json.loads(Path(out["manifest.json"]).read_text())["stale"] is True
    report = Path(out["report.md"]).read_text()
    assert "DESATUALIZADOS" in report and "desatualizados" in report


def test_report_contains_three_metric_groups_and_diagnostics(tmp_path):
    report = Path(export_results(make_result(), str(tmp_path))["report.md"]).read_text()
    for s in ("Desempenho preditivo", "Surrogate Fidelity", "Complexidade", "Diagnóstico das árvores", "TREPAN Original"):
        assert s in report
