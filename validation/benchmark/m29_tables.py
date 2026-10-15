"""Tabelas (markdown) do Doc. 30 geradas SÓ a partir do relatório M29 e dos brutos/auditorias/livros de CPU. Nada é arredondado antes de comparar; arredonda-se só na apresentação.
    python -m validation.benchmark.m29_tables --report FILE.json --m29 DIR [DIR ...] --ledgers FILE [FILE ...] --dry-run-dir DIR --out FILE.md"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ARMS = ["A29", "B29", "C29", "D29", "E29", "F29", "G29"]
NAMES = {"A29": "A29 Original+C", "B29": "B29 Reloaded OWL real+C", "C29": "C29 permutada+C", "D29": "D29 grupos aleatórios+C", "E29": "E29 convencional+C", "F29": "F29 Reloaded sem OWL+C",
         "G29": "G29 Original+C (orçamento igualado)"}
f4 = lambda x: "—" if x is None else f"{x:.4f}"
f1 = lambda x: "—" if x is None else f"{x:.1f}"


def md(rows, header):
    return "\n".join(["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def cost_table(ledgers):
    rows, tot = [], {"cpu": 0.0, "wall": 0.0}
    for lf in ledgers:
        recs = [json.loads(l) for l in Path(lf).read_text().splitlines() if l.strip()]
        cpu, wall = sum(r["cpu_s"] for r in recs), sum(r["wall_s"] for r in recs)
        tot["cpu"] += cpu
        tot["wall"] += wall
        rows.append([Path(lf).parent.name + "/" + Path(lf).name, len(recs), f"{cpu:.0f}", f"{wall:.0f}", f"{cpu / max(1e-9, wall):.2f}", f"{cpu / 3600:.3f}"])
    rows.append(["**TOTAL medido**", "", f"{tot['cpu']:.0f}", f"{tot['wall']:.0f}", f"{tot['cpu'] / max(1e-9, tot['wall']):.2f}", f"**{tot['cpu'] / 3600:.3f}**"])
    return md(rows, ["registo", "execuções", "CPU real (s)", "relógio (s)", "CPU/relógio", "CPU-h"]), tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", required=True)
    ap.add_argument("--m29", nargs="+", required=True)
    ap.add_argument("--ledgers", nargs="*", default=[])
    ap.add_argument("--dry-run-dir")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    r = json.loads(Path(a.report).read_text())
    out = []
    # --- braços (média das unidades)
    rows = []
    for arm in ARMS:
        v = r["arms"][arm]
        rows.append([NAMES[arm], v["n_units"], f4(v["accuracy"]), f4(v["fidelity"]), f4(v["precision_macro"]), f4(v["macro_f1"]), f1(v["node_count"]), f1(v["leaf_count"]), f1(v["tree_depth"]),
                     f1(v["m_of_n_split_count"]), f1(v["semantic_split_count"]), f1(v["oracle_queries"]), f1(v["oracle_samples"]), f1(v["training_time"])])
    out.append("### T3 — Métricas por braço (média das unidades; cada unidade = média das 3 réplicas/draws)\n" + md(rows, ["braço", "unid.", "accuracy", "fidelidade", "precision_macro", "macro-F1", "nós", "folhas", "profund.", "m-of-n", "splits semânticos", "consultas", "amostras do oráculo", "tempo (s)"]))
    # --- por dataset
    by = defaultdict(lambda: defaultdict(lambda: {"accuracy": [], "fidelity": [], "node_count": [], "oracle_queries": []}))
    for u, d in r["per_unit"].items():
        ds = u.split(":")[0]
        for arm in ARMS:
            v = d["vals"].get(arm)
            if v:
                for m in ("accuracy", "fidelity", "node_count", "oracle_queries"):
                    if v[m] is not None:
                        by[ds][arm][m].append(v[m])
    rows = []
    for ds in sorted(by):
        for arm in ARMS:
            if by[ds][arm]["accuracy"]:
                x = by[ds][arm]
                rows.append([ds, NAMES[arm], len(x["accuracy"]), f4(np.mean(x["accuracy"])), f4(np.std(x["accuracy"], ddof=1) if len(x["accuracy"]) > 1 else None), f4(np.mean(x["fidelity"])),
                             f4(np.std(x["fidelity"], ddof=1) if len(x["fidelity"]) > 1 else None), f1(np.mean(x["node_count"])), f1(np.mean(x["oracle_queries"]))])
    out.append("### T4 — Por dataset (média e DP entre as unidades/sementes mestras do dataset)\n" + md(rows, ["dataset", "braço", "unid.", "accuracy", "DP acc.", "fidelidade", "DP fid.", "nós", "consultas"]))
    # --- por dataset e réplica (draw k)
    rep = defaultdict(lambda: defaultdict(lambda: {"accuracy": [], "fidelity": []}))
    for u, d in r["per_unit"].items():
        ds = u.split(":")[0]
        for arm, lst in d["per_draw"].items():
            for k, x in enumerate(lst):
                if x["accuracy"] is not None:
                    rep[(ds, k)][arm]["accuracy"].append(x["accuracy"])
                    rep[(ds, k)][arm]["fidelity"].append(x["fidelity"])
    rows = []
    for (ds, k) in sorted(rep):
        rows.append([ds, k] + [f"{np.mean(rep[(ds, k)][arm]['accuracy']):.4f} / {np.mean(rep[(ds, k)][arm]['fidelity']):.4f}" if rep[(ds, k)][arm]["accuracy"] else "—" for arm in ARMS])
    out.append("### T5 — Por dataset e réplica/draw k (accuracy / fidelidade; média das sementes mestras)\n" + md(rows, ["dataset", "k"] + [x.split()[0] for x in ARMS]))
    # --- contrastes
    ct = r["contrast_tests"]
    rows = []
    for name in ("B29-A29", "B29-G29", "B29-C29", "B29-D29", "B29-E29"):
        c = ct.get(name, {})
        if not c.get("available"):
            rows.append([name, "—", "—", "—", "—", "—", "—"])
            continue
        d, u = c["dataset_level"], c["unit_level"]
        rows.append([name, f"{d['accuracy']['mean']:+.4f} (EP {f4(d['accuracy']['se'])})", f"{d['fidelity']['mean']:+.4f} (EP {f4(d['fidelity']['se'])})",
                     f"{u['accuracy']['mean']:+.4f} (EP {f4(u['accuracy']['se'])})", f"{u['fidelity']['mean']:+.4f} (EP {f4(u['fidelity']['se'])})", f4(c.get("p_IUT_dataset")), f4(c.get("p_IUT_holm_family"))])
    out.append("### T6 — Contrastes de B29 (Δ = B29 − braço; nível DATASET n=4 e nível UNIDADE descritivo)\n" + md(rows, ["contraste", "Δ accuracy (datasets)", "Δ fidelidade (datasets)", "Δ accuracy (unidades)", "Δ fidelidade (unidades)", "p_IUT (datasets)", "p_IUT Holm"]))
    ev = r["level1"]
    rows = [["B29 vs A29 (Original equivalente)", ev["evidence_vs_original_equivalent"]["evidence"], f4(ev["evidence_vs_original_equivalent"]["across_datasets"]["acc"]["mean"]),
             str([round(x, 4) for x in ev["evidence_vs_original_equivalent"]["across_datasets"]["acc"]["bootstrap_ci"]]), f4(ev["evidence_vs_original_equivalent"]["across_datasets"]["fid"]["mean"]),
             str([round(x, 4) for x in ev["evidence_vs_original_equivalent"]["across_datasets"]["fid"]["bootstrap_ci"]])]]
    for c, e in ev["evidence_vs_controls"].items():
        if "across_datasets" in e:
            rows.append([f"B29 vs controlo {c}", e["evidence"], f4(e["across_datasets"]["acc"]["mean"]), str([round(x, 4) for x in e["across_datasets"]["acc"]["bootstrap_ci"]]),
                         f4(e["across_datasets"]["fid"]["mean"]), str([round(x, 4) for x in e["across_datasets"]["fid"]["bootstrap_ci"]])])
    out.append("### T7 — Evidência entre datasets (unidade de inferência = dataset; exploratório)\n" + md(rows, ["contraste", "nível de evidência", "Δ accuracy", "IC95 bootstrap", "Δ fidelidade", "IC95 bootstrap"]))
    out.append("### T8 — Estados lógicos por unidade (TODAS as unidades)\n" + md([[k, v["n"], f"{v['pct_of_all_units']}%"] for k, v in ev["direct_gain_state_counts"].items()], ["Nível 1: ganho direto", "n", "%"]) + "\n\n"
               + md([[k, v] for k, v in ev["attribution_counts"].items()], ["atribuição ontológica (completa)", "n"]) + "\n\n" + md([[k, v] for k, v in ev["attribution_counts_excluding_E29"].items()], ["atribuição sem E29 (secundária, não conclusiva)", "n"])
               + "\n\n" + md([[k, v["n"], f"{v['pct_of_all_units']}%"] for k, v in r["level2"]["global_state_counts"].items()], ["Nível 2: hierarquia global", "n", "%"]))
    # --- confusão por controlo
    cf = defaultdict(Counter)
    for u, d in r["per_unit"].items():
        for arm, x in ((d.get("states") or {}).get("control_confounding") or {}).items():
            cf[arm]["confundido" if x["confounded"] else "comparável"] += 1
    out.append("### T9 — Confusão de dimensão/orçamento por controlo (nº de unidades)\n" + md([[NAMES[a], c.get("comparável", 0), c.get("confundido", 0)] for a, c in sorted(cf.items())], ["controlo", "comparável", "confundido"]))
    # --- mecanismos
    mech = r.get("mechanism_aggregates", {})
    out.append("### T10 — Agregados de mecanismo ontológico por árvore (média)\n" + md([[NAMES[a]] + [f1(mech[a].get(k)) for k in ("ontology_influenced_splits", "semantic_decision_changed_count", "semantic_rule_count", "semantic_features_used", "semantic_split_count", "m_of_n_count", "internal_nodes")]
                                                                                       for a in ARMS if a in mech], ["braço", "splits influenciados", "decisões alteradas", "regras semânticas", "features semânticas usadas", "splits semânticos", "m-of-n", "nós internos"]))
    # --- cost
    if a.ledgers:
        tbl, tot = cost_table(a.ledgers)
        out.append("### T1 — Custo efetivo (CPU real por processo, `os.wait4`)\n" + tbl)
    Path(a.out).write_text("\n\n".join(out) + "\n")
    print("ok", len(out), "tabelas")


if __name__ == "__main__":
    main()
