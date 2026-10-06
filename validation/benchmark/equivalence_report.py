"""BASELINE_VS_OPTIMIZED_EQUIVALENCE: PASS/FAIL por campo e por dataset entre uma execução de referência e uma otimizada.

A otimização só é válida se preservar TODOS os resultados científicos (comparação EXATA; só tempos e contadores de trabalho são ignorados).
    python -m validation.benchmark.equivalence_report --pair iris BASE_ROOT OPT_ROOT --pair wine BASE_ROOT OPT_ROOT --out DIR
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd

from core.benchmark import analysis as an
from validation.benchmark import equivalence as eq
from validation.benchmark.invariants import GATE_FIELDS
from validation.benchmark.raw_store import load_raw

ARMS = {"MLP (A)": "mlp_original", "C4.5 (B)": "c45", "TREPAN Original (C)": "trepan_original", "Reloaded Core (D)": "reloaded_core",
        "Reloaded + OWL (E)": "reloaded_owl_full", "Reloaded + OWL shuffled (F)": "reloaded_owl_shuffled"}
METRICS = {"fidelity": "fidelity_to_oracle", "accuracy": "accuracy_real_labels", "balanced accuracy": "balanced_accuracy_real_labels",
           "macro-F1": "macro_f1_real_labels", "nós": "node_count", "folhas": "leaf_count", "profundidade": "depth"}
SEMANTIC_AUDIT = ("semantic_split_count", "ontology_influenced_splits", "semantic_decision_changed_count", "ontology_usage_rate",
                  "semantic_decision_impact", "semantic_rule_count", "semantic_features_used")


def _same(a: Any, b: Any) -> bool:
    return eq._equal(a, b, 0.0)


def _by_arm(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {r["arm"]: r for r in records}


def compare_dataset(root_a: Path, root_b: Path, dataset: str) -> List[Dict[str, Any]]:
    rows_a, recs_a, splits_a = load_raw(root_a)
    rows_b, recs_b, splits_b = load_raw(root_b)
    rows_a, rows_b = rows_a[rows_a["dataset"] == dataset], rows_b[rows_b["dataset"] == dataset]
    ra = _by_arm([r for r in recs_a if r["dataset"] == dataset])
    rb = _by_arm([r for r in recs_b if r["dataset"] == dataset])
    sa = next(s for s in splits_a if s["dataset"] == dataset)["semantic_report"][0]
    sb = next(s for s in splits_b if s["dataset"] == dataset)["semantic_report"][0]
    out: List[Dict[str, Any]] = []

    def add(field: str, ok: bool, detail: str = "") -> None:
        out.append({"dataset": dataset, "field": field, "status": "PASS" if ok else "FAIL", "detail": "" if ok else detail})

    common = sorted(set(ra) & set(rb))
    add("conjunto de braços A–F", set(ra) == set(rb) and set(ARMS.values()) <= set(ra), f"{sorted(set(ra) ^ set(rb))}")
    add("split_hash", all(ra[a]["split_hash"] == rb[a]["split_hash"] for a in common), "split_hash difere")
    add("preprocessing_id", all(ra[a]["preprocessing_id"] == rb[a]["preprocessing_id"] for a in common), "preprocessing_id difere")
    add("oracle_id", all(ra[a]["oracle_id"] == rb[a]["oracle_id"] and ra[a]["fidelity_oracle_id"] == rb[a]["fidelity_oracle_id"] for a in common),
        "oracle_id difere")
    pa, pb = sa["structural_protocol"], sb["structural_protocol"]
    strip = lambda p: eq._strip_time({k: v for k, v in p.items() if k not in ("engineering_profile", "number_cv_fits", "number_capacity_fits")})
    add("tuning config selecionada (estrutural)", _same(strip(pa), strip(pb)), "relatório do tuning estrutural difere")
    for key in ("purity_epsilon", "max_nodes", "max_queries"):
        add({"max_queries": "query_budget"}.get(key, key), _same((pa.get("selected") or {}).get(key), (pb.get("selected") or {}).get(key)),
            f"{(pa.get('selected') or {}).get(key)} != {(pb.get('selected') or {}).get(key)}")
    for label, arm in ARMS.items():
        if arm in ra and arm in rb:
            pr_a, pr_b = ra[arm]["predictions"], rb[arm]["predictions"]
            same = all(_same(pr_a[k], pr_b[k]) for k in ("test_row_index", "y_true", "oracle_predictions", "surrogate_predictions"))
            add(f"predictions {label}", same, "predições diferentes")
    for name, col in METRICS.items():
        bad = [a for a in common if not _same(eq._strip_time(ra[a]["row"]).get(col), eq._strip_time(rb[a]["row"]).get(col))]
        add(name, not bad, f"difere em {bad}")
    gate_arms = [a for a in ("reloaded_core", "reloaded_owl_full", "reloaded_owl_shuffled") if a in common]
    add("quality gate", all(_same(ra[a]["row"].get(f), rb[a]["row"].get(f)) for a in gate_arms for f in GATE_FIELDS), "campos do gate diferem")
    add("semantic audit", all(_same(ra[a]["row"].get(f), rb[a]["row"].get(f)) for a in gate_arms for f in SEMANTIC_AUDIT)
        and _same(eq._strip_time({k: v for k, v in sa.items() if k not in ("work", "timing", "structural_protocol", "oracle_contract", "tuning_execution_count")}),
                  eq._strip_time({k: v for k, v in sb.items() if k not in ("work", "timing", "structural_protocol", "oracle_contract", "tuning_execution_count")})),
        "auditoria semântica difere")
    add("ontology_effectively_used / categoria", all(_same(ra[a]["row"].get("ontology_effectively_used"), rb[a]["row"].get("ontology_effectively_used"))
                                                        and _same(ra[a]["row"].get("ontology_category"), rb[a]["row"].get("ontology_category")) for a in gate_arms),
        "ontology_effectively_used difere")
    add("estrutura das árvores (features/splits)", all(_same(ra[a]["row"].get(c), rb[a]["row"].get(c)) for a in common
                                                        for c in ("features_used_names", "root_feature", "split_signatures", "rule_count")), "estrutura difere")
    ca = an.run_contrasts(rows_a, scheme="holdout", n_boot=200, seed=0)
    cb = an.run_contrasts(rows_b, scheme="holdout", n_boot=200, seed=0)
    cols = [c for c in ("contrast", "metric", "status", "n_pairs", "mean_a", "mean_b", "mean_diff", "median_diff", "wins_a", "wins_b", "ties",
                        "wilcoxon_p", "p_holm", "evidence_level") if c in ca and c in cb]
    key = lambda d: d[cols].sort_values(["contrast", "metric"]).reset_index(drop=True)
    same_contrasts = len(ca) == len(cb) and eq.compare_frames(key(ca).assign(split_id="x", arm=key(ca)["contrast"] + "/" + key(ca)["metric"]),
                                                              key(cb).assign(split_id="x", arm=key(cb)["contrast"] + "/" + key(cb)["metric"])) == []
    add("contrastes", same_contrasts, "contrastes diferem")
    return out


def build_report(pairs: List[Tuple[str, Path, Path]]) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    for dataset, a, b in pairs:
        results += compare_dataset(Path(a), Path(b), dataset)
    return {"title": "BASELINE_VS_OPTIMIZED_EQUIVALENCE", "datasets": sorted({r["dataset"] for r in results}),
            "n_fields": len(results), "n_fail": sum(r["status"] == "FAIL" for r in results),
            "identical": all(r["status"] == "PASS" for r in results) and bool(results), "tolerance": 0.0,
            "ignored": "tempos, contadores de trabalho (queries por escopo do tuning, número de ajustes) e identificadores de execução", "results": results}


def to_markdown(report: Dict[str, Any]) -> str:
    df = pd.DataFrame(report["results"])
    table = df.pivot_table(index="field", columns="dataset", values="status", aggfunc="first", sort=False)
    lines = ["# BASELINE_VS_OPTIMIZED_EQUIVALENCE\n", f"Datasets: {', '.join(report['datasets'])} · campos comparados: {report['n_fields']} · FAIL: {report['n_fail']} · "
             f"tolerância numérica: {report['tolerance']} (comparação exata)\n", f"**Resultado global: {'PASS' if report['identical'] else 'FAIL'}**\n",
             "| campo | " + " | ".join(table.columns) + " |", "|---|" + "|".join("---" for _ in table.columns) + "|"]
    for field, row in table.iterrows():
        lines.append(f"| {field} | " + " | ".join(str(row[c]) for c in table.columns) + " |")
    fails = df[df["status"] == "FAIL"]
    if len(fails):
        lines += ["\n## Falhas\n"] + [f"- {r.dataset} / {r.field}: {r.detail}" for r in fails.itertuples()]
    lines.append(f"\n_Ignorado: {report['ignored']}._\n")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pair", nargs=3, action="append", metavar=("DATASET", "BASE_ROOT", "OPT_ROOT"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rep = build_report([(d, Path(b), Path(o)) for d, b, o in a.pair])
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (out / "EQUIVALENCE_REPORT_v1_to_v2.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    (out / "BASELINE_VS_OPTIMIZED_EQUIVALENCE.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))
    raise SystemExit(0 if rep["identical"] else 1)
