"""Exportação determinística de resultados contrafactuais."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

import numpy as np


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(item) for key, item in value.items()
            if not str(key).startswith("_runtime_")
        }
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def _flatten(prefix: str, value: Mapping[str, Any], output: Dict[str, Any]) -> None:
    for key, item in value.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            _flatten(name, item, output)
        elif isinstance(item, (list, tuple, dict)):
            output[name] = json.dumps(_json_safe(item), ensure_ascii=False)
        else:
            output[name] = _json_safe(item)


def export_counterfactual_result(
    result: Mapping[str, Any],
    directory: Any,
    *,
    stem: str = "counterfactual_report",
) -> Dict[str, str]:
    """Exporta JSON, CSV e Markdown; devolve os caminhos criados."""
    output_dir = Path(directory)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = _json_safe(result)
    json_path = output_dir / f"{stem}.json"
    csv_path = output_dir / f"{stem}.csv"
    markdown_path = output_dir / f"{stem}.md"
    json_path.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    records: Iterable[Mapping[str, Any]]
    if safe.get("rows") is not None:
        records = safe.get("rows") or []
    elif safe.get("counterfactual_rules") is not None:
        records = safe.get("counterfactual_rules") or []
    elif safe.get("tree_rules") is not None:
        records = safe.get("tree_rules") or []
    else:
        records = safe.get("candidates") or []
    flat_rows = []
    for record in records:
        flat: Dict[str, Any] = {}
        _flatten("", record, flat)
        flat_rows.append(flat)
    columns = sorted({key for row in flat_rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        if columns:
            writer.writeheader()
            writer.writerows(flat_rows)

    markdown_path.write_text(_to_markdown(safe), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(markdown_path),
    }


def _to_markdown(result: Mapping[str, Any]) -> str:
    lines = ["# Relatório contrafactual", ""]
    if result.get("narrative"):
        lines.extend([str(result["narrative"]), ""])
    if result.get("interpretation"):
        lines.extend([str(result["interpretation"]), ""])
    summary = result.get("aggregate_metrics") or result.get("summary") or {}
    if summary:
        lines.extend(["## Métricas", "", "| Métrica | Valor |", "|---|---:|"])
        for key, value in summary.items():
            if isinstance(value, (Mapping, list, tuple)):
                continue
            if isinstance(value, float):
                display = f"{value:.6f}"
            else:
                display = str(value)
            lines.append(f"| {key} | {display} |")
        lines.append("")

    candidates = result.get("candidates") or []
    if candidates:
        lines.extend(["## Candidatos", ""])
        for index, candidate in enumerate(candidates, 1):
            lines.append(f"### {index}. {candidate.get('method', 'método')}")
            lines.append("")
            lines.append(candidate.get("rule") or "Sem regra disponível.")
            lines.append("")
    rows = result.get("rows") or []
    if rows:
        lines.extend([
            f"## Transferência no dataset carregado: {result.get('dataset', 'actual')}",
            "",
            "| Instância | Método | Categoria | Robustez conjunta |",
            "|---:|---|---|---:|",
        ])
        for row in rows:
            joint = row.get("joint_robustness")
            joint_text = f"{joint:.4f}" if isinstance(joint, (int, float)) else "N/A"
            lines.append(
                f"| {row.get('instance_index', '')} | {row.get('method', '')} | "
                f"{row.get('category', '')} | {joint_text} |"
            )
        lines.append("")
    global_rules = result.get("counterfactual_rules") or []
    if global_rules:
        lines.extend([
            f"## Regras contrafactuais globais — {result.get('target_model', 'árvore')}",
            "",
            "| Regra factual | Regra destino | Classe | Alterações | Confiança |",
            "|---|---|---|---:|---:|",
        ])
        for row in global_rules:
            confidence = row.get("target_confidence")
            confidence_text = f"{confidence:.4f}" if isinstance(confidence, (int, float)) else "N/A"
            lines.append(
                f"| {row.get('factual_rule_id', '')} | "
                f"{row.get('counterfactual_rule_id', '')} | "
                f"{row.get('target_class_name', row.get('target_class', ''))} | "
                f"{row.get('n_changes', '')} | {confidence_text} |"
            )
        lines.append("")
    tree_rules = result.get("tree_rules") or []
    if tree_rules:
        lines.extend(["## Árvore explicativa contrafactual", ""])
        for row in tree_rules:
            lines.extend([
                f"### {row.get('rule_id', 'Regra')}", "",
                str(row.get("rule") or "Sem descrição."), "",
            ])
    formal = result.get("formal_evaluation") or {}
    if formal.get("summary"):
        lines.extend(["## Avaliação formal", "", "| Métrica | Valor |", "|---|---:|"])
        for key, value in formal["summary"].items():
            display = f"{value:.6f}" if isinstance(value, float) else str(value)
            lines.append(f"| {key} | {display} |")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


__all__ = ["export_counterfactual_result"]
