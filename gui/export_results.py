"""Exportação estruturada de resultados (Partes 35-36). Qt-free.

"Exportar resultados" escreve **dados e relatório** (métricas, relatório semântico, diagnóstico das
árvores, manifesto da experiência, configuração, resumo contrafactual). A exportação da árvore como
imagem/SVG/PDF é uma ação distinta (``BiuriApp.export_tree``) e não passa por aqui.
Nada é recalculado: tudo vem do ``ExperimentResult``.
"""
from __future__ import annotations

import csv
import json
import os
import time
from typing import Dict, List

from core.experiment_result import ExperimentResult, Reason
from gui import result_presenter as rp
from gui.strings import term, tr

FILES = ("results.json", "manifest.json", "config.json", "metrics.csv", "semantic_features.csv", "semantic_splits.csv",
         "tree_diagnostics.json", "messages.json", "report.md")


def _dump(path: str, obj) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, default=str)


def _csv(path: str, headers: List[str], rows: List[List]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        w.writerows(rows)


def metrics_rows(result: ExperimentResult) -> List[List]:
    """Linhas planas model/group/metric/value/reason/oracle (valor vazio + razão quando não existe)."""
    rows: List[List] = []

    def add(model, group, metric, measure, oracle=""):
        rows.append([model, group, metric, "" if measure is None or measure.value is None else measure.value,
                     "" if measure is not None and measure.value is not None else (measure.reason if measure else Reason.NOT_REPORTED),
                     oracle])

    for key, card in result.models.items():
        oracle = card.oracle or ""
        for m, v in card.metrics.items():
            add(key, "predictive", m, v)
        if key.startswith("trepan") or key == "c45":
            add(key, "fidelity", "fidelity", card.fidelity, oracle)
        if key == "c45":
            add(key, "diagnostic", "agreement_with_mlp", card.agreement_with_mlp)
        for m, v in card.complexity.items():
            add(key, "complexity", m, v)
    return rows


def render_report(result: ExperimentResult) -> str:
    lines = [f"# {tr('section.experiment')} {result.provenance.experiment_id}", ""]
    banner = rp.stale_banner(result)
    if banner:
        lines += [f"> **{banner}**", "", f"> {tr('export.stale_warning')}", ""]
    cache = rp.cache_banner(result)
    if cache:
        lines += [f"> {cache}", ""]
    sections = [(tr("tab.summary"), rp.summary_rows(result)), (tr("section.dataset"), rp.dataset_rows(result)),
                (tr("section.ontology"), rp.ontology_rows(result)), (tr("section.enrichment"), rp.enrichment_rows(result)),
                (tr("section.experiment"), rp.experiment_rows(result))]
    for title, rows in sections:
        lines += [f"## {title}", ""] + [f"- **{k}**: {v}" for k, v in rows] + [""]
    tables = rp.metrics_tables(result)
    for group, title_key in (("predictive", "section.metrics.predictive"), ("fidelity", "section.metrics.fidelity"),
                             ("complexity", "section.metrics.complexity")):
        t = tables[group]
        lines += [f"## {tr(title_key)}", "", "| " + " | ".join(t["headers"]) + " |", "|" + "---|" * len(t["headers"])]
        lines += ["| " + " | ".join(str(c) for c in row) + " |" for row in t["rows"]] + [""]
    lines += [f"## {tr('section.diagnostics')}", ""]
    for key, diag in result.trees.items():
        lines += [f"### {term(key)}"] + [f"- **{k}**: {v}" for k, v in rp.tree_diagnostic_rows(diag)] + [""]
    if result.messages:
        lines += [f"## {tr('section.log')}", ""] + [f"- [{tr('level.' + m.level)}] {m.text}" for m in result.messages] + [""]
    return "\n".join(lines)


def export_results(result: ExperimentResult, directory: str, *, counterfactual_summary=None) -> Dict[str, str]:
    """Escreve os ficheiros estruturados em ``directory`` e devolve ``{nome: caminho}``."""
    os.makedirs(directory, exist_ok=True)
    out: Dict[str, str] = {}

    def path(name: str) -> str:
        out[name] = os.path.join(directory, name)
        return out[name]

    data = result.to_dict()
    data["exported_at"] = time.time()
    _dump(path("results.json"), data)
    _dump(path("manifest.json"), {"experiment_id": result.provenance.experiment_id, "provenance": data["provenance"],
                                  "dataset": data["dataset"], "ontology": data["ontology"], "stale": result.stale,
                                  "stale_reasons": result.stale_reasons, "state": result.state,
                                  "schema_version": result.schema_version, "files": list(FILES)})
    _dump(path("config.json"), {"config": result.config, "config_hash": result.provenance.config_hash,
                                "seed": result.provenance.seed})
    _csv(path("metrics.csv"), ["model", "group", "metric", "value", "reason", "oracle"], metrics_rows(result))
    feats = rp.semantic_features_table(result)
    _csv(path("semantic_features.csv"), feats["headers"], feats["rows"])
    splits = rp.semantic_splits_table(result)
    _csv(path("semantic_splits.csv"), splits["headers"], splits["rows"])
    _dump(path("tree_diagnostics.json"), data["trees"])
    _dump(path("messages.json"), data["messages"])
    if counterfactual_summary is not None or result.counterfactual:
        _dump(path("counterfactual_summary.json"), counterfactual_summary if counterfactual_summary is not None else result.counterfactual)
    with open(path("report.md"), "w", encoding="utf-8") as fh:
        fh.write(render_report(result))
    return out
