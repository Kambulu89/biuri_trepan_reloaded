"""B3.2 (Doc. 38): análise dos resultados congelados (E1/E2, estabilidade E3, memória) segundo os critérios PRÉ-REGISTADOS (b32_frozen.json). Não altera critérios."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from validation.benchmark.b31_identifiability_report import mcnemar_exact, wilson

PERF = ("recovered", "fidelity", "accuracy", "precision_macro", "macro_f1")
RES = ("internal_nodes", "depth", "mean_rule_literals", "literals_listed", "eval_calls", "simple_generated", "mofn_generated", "mofn_evaluated", "queries", "oracle_samples",
       "cpu_fit_s", "compile_wall_s", "n_units", "unit_searches", "choice_changed")


def boot(x, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float)
    b = [float(np.mean(rng.choice(x, len(x)))) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def by_arm(rows):
    d = defaultdict(dict)
    for r in rows:
        d[r["arm"]][r["seed"]] = r
    return d


def analyse_e(res: Dict[str, Any]) -> Dict[str, Any]:
    by = by_arm(res["rows"])
    refs = {r["seed"]: r for r in res["refs"]}
    seeds = sorted(by["A_ORIG"])
    out: Dict[str, Any] = {"n": len(seeds), "arms": {}, "refs": {}}
    for a, d in by.items():
        t: Dict[str, Any] = {}
        rec = [d[s]["recovered"] for s in seeds]
        t["recovery"] = float(np.mean(rec)); t["recovery_ci95"] = list(wilson(sum(rec), len(rec)))
        for k in PERF[1:] + RES:
            v = [d[s][k] for s in seeds if d[s].get(k) is not None]
            t[k] = float(np.mean(v)) if v else None
            if k in ("fidelity", "accuracy", "internal_nodes"):
                t[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else None
        out["arms"][a] = t
    out["refs"] = {"C45_accuracy": float(np.mean([refs[s]["accuracy"] for s in seeds])), "C45_fidelity": float(np.mean([refs[s]["fidelity"] for s in seeds])),
                   "C45_precision_macro": float(np.mean([refs[s]["precision_macro"] for s in seeds])), "C45_macro_f1": float(np.mean([refs[s]["macro_f1"] for s in seeds])),
                   "MLP_accuracy": float(np.mean([refs[s]["mlp_accuracy"] for s in seeds]))}
    # contrastes de C contra cada braço
    cmp_: Dict[str, Any] = {}
    C = by["C_OWL"]
    for a, d in by.items():
        if a == "C_OWL":
            continue
        dfid = [C[s]["fidelity"] - d[s]["fidelity"] for s in seeds]
        dacc = [C[s]["accuracy"] - d[s]["accuracy"] for s in seeds]
        b_ = sum(1 for s in seeds if C[s]["recovered"] and not d[s]["recovered"])
        c_ = sum(1 for s in seeds if d[s]["recovered"] and not C[s]["recovered"])
        cmp_[a] = {"d_recovery": out["arms"]["C_OWL"]["recovery"] - out["arms"][a]["recovery"], "mcnemar_b_c": [b_, c_], "mcnemar_p": mcnemar_exact(b_, c_),
                   "d_fidelity": float(np.mean(dfid)), "d_fidelity_ci95": list(boot(dfid)), "d_accuracy": float(np.mean(dacc)), "d_accuracy_ci95": list(boot(dacc)),
                   "ratio_eval_calls": out["arms"]["C_OWL"]["eval_calls"] / max(out["arms"][a]["eval_calls"], 1e-9), "ratio_cpu_fit": out["arms"]["C_OWL"]["cpu_fit_s"] / max(out["arms"][a]["cpu_fit_s"], 1e-9),
                   "ratio_queries": out["arms"]["C_OWL"]["queries"] / max(out["arms"][a]["queries"], 1e-9), "ratio_internal_nodes": out["arms"]["C_OWL"]["internal_nodes"] / max(out["arms"][a]["internal_nodes"], 1e-9),
                   "ratio_fidelity_sd": (out["arms"]["C_OWL"]["fidelity_sd"] or 0) / max(out["arms"][a]["fidelity_sd"] or 1e-9, 1e-9)}
    out["C_vs"] = cmp_
    A = out["arms"]
    rec = lambda a: A[a]["recovery"]
    o1 = rec("C_OWL") - max(rec(x) for x in ("D_PERM", "E_RAND", "F1_CONV", "F2_SCREEN", "G_NEUTRALIZED")) >= 0.20
    def o_gen(a):
        c = cmp_[a]
        return bool((c["d_fidelity"] >= 0.005 and c["d_fidelity_ci95"][0] > 0) or c["d_recovery"] >= 0.10)
    b = cmp_["B_WIDE"]
    o4a = b["d_fidelity"] >= 0.005 and b["d_fidelity_ci95"][0] > 0
    o4b = b["ratio_cpu_fit"] <= 0.8 and rec("C_OWL") >= rec("B_WIDE") - 0.05
    o4c = b["ratio_internal_nodes"] <= 0.8 and b["d_fidelity"] >= -0.005
    o4d = b["ratio_fidelity_sd"] <= 0.8 and b["d_fidelity"] >= -0.005
    crit = {"O1_relation_specific": bool(o1), "O2_vs_generic_equal_candidates(H1)": o_gen("H1_WIDE_BEAM_CAND"), "O3_vs_generic_equal_cpu(H2)": o_gen("H2_WIDE_BEAM_CPU"),
            "O4_vs_WIDE": {"a_fidelity": bool(o4a), "b_cost": bool(o4b), "c_complexity": bool(o4c), "d_stability": bool(o4d), "any": bool(o4a or o4b or o4c or o4d)}}
    allp = crit["O2_vs_generic_equal_candidates(H1)"] and crit["O3_vs_generic_equal_cpu(H2)"] and crit["O4_vs_WIDE"]["any"]
    crit["verdict"] = "ONTOLOGICAL_ADVANTAGE_OVER_GENERIC" if (o1 and allp) else ("RELATION_SPECIFIC_ONLY" if o1 else "NO_ONTOLOGICAL_ADVANTAGE")
    out["criteria"] = crit
    # hierarquia exploratória (Original+orientação como proxy do Reloaded)
    pa = lambda x, y: float(np.mean([by[x][s]["accuracy"] > by[y][s]["accuracy"] for s in seeds]))
    out["hierarchy_exploratory"] = {"mean_Acc_C>Acc_A": A["C_OWL"]["accuracy"] > A["A_ORIG"]["accuracy"], "mean_Acc_A>Acc_C45": A["A_ORIG"]["accuracy"] > out["refs"]["C45_accuracy"],
                                    "mean_Fid_C>Fid_A": A["C_OWL"]["fidelity"] > A["A_ORIG"]["fidelity"], "mean_Fid_C>Fid_C45": A["C_OWL"]["fidelity"] > out["refs"]["C45_fidelity"],
                                    "per_seed_rate_Acc_C>A": pa("C_OWL", "A_ORIG"), "per_seed_rate_Fid_C>A": float(np.mean([by["C_OWL"][s]["fidelity"] > by["A_ORIG"][s]["fidelity"] for s in seeds])),
                                    "all_five_conditions_means": bool(A["C_OWL"]["accuracy"] > A["A_ORIG"]["accuracy"] > out["refs"]["C45_accuracy"] and A["C_OWL"]["fidelity"] > A["A_ORIG"]["fidelity"] and A["C_OWL"]["fidelity"] > out["refs"]["C45_fidelity"])}
    return out


def md_e(name, an):
    L = [f"### {name} (n = {an['n']} sementes)\n", "| braço | recup. [IC95] | fidelidade (±DP) | accuracy (±DP) | prec. macro | macro-F1 | nós int. | prof. | lit./regra | literais listados | cand. gerados (simples+m-of-n) | cand. AVALIADOS | consultas | amostras ao oráculo | CPU ajuste (s) | compil. (s) | unid. pesquisadas |", "|" + "---|" * 17]
    for a, t in an["arms"].items():
        gen = (t["simple_generated"] or 0) + (t["mofn_generated"] or 0)
        L.append(f"| {a} | {t['recovery']:.3f} [{t['recovery_ci95'][0]:.2f}; {t['recovery_ci95'][1]:.2f}] | {t['fidelity']:.4f} ± {t['fidelity_sd']:.4f} | {t['accuracy']:.4f} ± {t['accuracy_sd']:.4f} | {t['precision_macro']:.4f} | {t['macro_f1']:.4f} | {t['internal_nodes']:.2f} | "
                 f"{t['depth']:.2f} | {t['mean_rule_literals']:.2f} | {t['literals_listed']:.0f} | {gen:.0f} | {t['eval_calls']:.0f} | {t['queries']:.0f} | {t['oracle_samples']:.0f} | {t['cpu_fit_s']:.2f} | {t['compile_wall_s']:.2f} | {t['unit_searches']:.1f} |")
    r = an["refs"]
    L.append(f"\nReferências: MLP accuracy {r['MLP_accuracy']:.4f}; C4.5 canónico accuracy {r['C45_accuracy']:.4f}, fidelidade ao MLP {r['C45_fidelity']:.4f}, precision macro {r['C45_precision_macro']:.4f}, macro-F1 {r['C45_macro_f1']:.4f}.\n")
    L.append("| C vs … | Δ recup. | McNemar (b,c; p) | Δ fidelidade [IC95] | Δ accuracy [IC95] | cand. avaliados C/x | CPU C/x | consultas C/x | nós int. C/x |\n|---|---|---|---|---|---|---|---|---|")
    for a, c in an["C_vs"].items():
        L.append(f"| {a} | {c['d_recovery']:+.3f} | ({c['mcnemar_b_c'][0]},{c['mcnemar_b_c'][1]}); {c['mcnemar_p']:.3f} | {c['d_fidelity']:+.4f} [{c['d_fidelity_ci95'][0]:+.4f}; {c['d_fidelity_ci95'][1]:+.4f}] | {c['d_accuracy']:+.4f} [{c['d_accuracy_ci95'][0]:+.4f}; {c['d_accuracy_ci95'][1]:+.4f}] | {c['ratio_eval_calls']:.2f} | {c['ratio_cpu_fit']:.2f} | {c['ratio_queries']:.2f} | {c['ratio_internal_nodes']:.2f} |")
    cr = an["criteria"]
    L.append(f"\nCritérios pré-registados: O1={cr['O1_relation_specific']}; O2(H1)={cr['O2_vs_generic_equal_candidates(H1)']}; O3(H2)={cr['O3_vs_generic_equal_cpu(H2)']}; O4(vs WIDE)={cr['O4_vs_WIDE']}; **veredicto: {cr['verdict']}**")
    L.append(f"\nHierarquia exploratória: {an['hierarchy_exploratory']}\n")
    return "\n".join(L)


def analyse_stability(res):
    rows = res["rows"]
    out = {}
    for ci, cfg in enumerate(res["configs"]):
        rr = [r for r in rows if r["config"] == ci]
        by = by_arm(rr)
        out[ci] = {"config": cfg, "recovery": {a: float(np.mean([r["recovered"] for r in d.values()])) for a, d in by.items()},
                   "fidelity": {a: float(np.mean([r["fidelity"] for r in d.values()])) for a, d in by.items()}, "eval_calls": {a: float(np.mean([r["eval_calls"] for r in d.values()])) for a, d in by.items()}}
    return out


def analyse_memory(res):
    by = by_arm(res["rows"])
    return {a: float(np.mean([r["peak_memory_bytes"] for r in d.values()]) / 1e6) for a, d in by.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e", nargs="*", default=[]); ap.add_argument("--stability"); ap.add_argument("--memory", nargs="*", default=[]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res: Dict[str, Any] = {"e": {}, "memory": {}}
    md = []
    for f in a.e:
        r = json.loads(Path(f).read_text())
        an = analyse_e(r)
        res["e"][r["problem"]] = an
        md.append(md_e(r["problem"], an))
    if a.stability:
        res["stability"] = analyse_stability(json.loads(Path(a.stability).read_text()))
    for f in a.memory:
        r = json.loads(Path(f).read_text())
        res["memory"][r["problem"]] = analyse_memory(r)
    Path(a.out).write_text(json.dumps(res, indent=1))
    Path(a.out).with_suffix(".md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
