"""B3.1 (Doc. 36): relatório do avaliador congelado — taxas de recuperação (IC de Wilson), fidelidade independente, contrastes emparelhados (McNemar exato), custo e
avaliação dos critérios PRÉ-REGISTADOS (problems_frozen.json). Não altera critérios; reporta cumprimento/incumprimento."""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def analyse(res):
    out = {}
    for pname, rows in res["problems"].items():
        by = defaultdict(dict)
        for r in rows:
            by[r["variant"]][r["seed"]] = r
        seeds = sorted(by["ORIG"])
        tab = {}
        for v, d in by.items():
            rec = [d[s]["recovered"] for s in seeds]
            fid = [d[s]["fidelity_test"] for s in seeds]
            lo, hi = wilson(sum(rec), len(rec))
            tab[v] = {"n": len(seeds), "recovery": float(np.mean(rec)), "recovery_ci95": [lo, hi], "fidelity_mean": float(np.mean(fid)), "fidelity_sd": float(np.std(fid, ddof=1)),
                      "time_ratio_vs_orig": float(np.mean([d[s]["time_s"] for s in seeds]) / np.mean([by["ORIG"][s]["time_s"] for s in seeds])),
                      "queries_mean": float(np.mean([d[s]["queries"] for s in seeds])), "choice_changed_mean": float(np.mean([d[s]["n_choice_changed"] for s in seeds])),
                      "influenced_in_final_tree_mean": float(np.mean([d[s]["n_influenced_in_final_tree"] for s in seeds])), "units_mean": float(np.mean([d[s]["n_units"] for s in seeds])),
                      "identical_tree_to_orig": int(sum(d[s]["rules_sha"] == by["ORIG"][s]["rules_sha"] for s in seeds))}
            if v != "ORIG":
                a = [d[s]["recovered"] for s in seeds]
                o = [by["ORIG"][s]["recovered"] for s in seeds]
                b_ = sum(1 for x, y in zip(a, o) if x and not y)
                c_ = sum(1 for x, y in zip(a, o) if y and not x)
                tab[v].update(delta_recovery_vs_orig=float(np.mean(a) - np.mean(o)), mcnemar_b_c=[b_, c_], mcnemar_p=mcnemar_exact(b_, c_),
                              delta_fidelity_vs_orig=float(np.mean(fid) - np.mean([by["ORIG"][s]["fidelity_test"] for s in seeds])))
        out[pname] = {"table": tab, "n_seeds": len(seeds)}
        crit = {}
        R = lambda v: tab[v]["recovery"]
        if pname == "P1_within_group":
            crit["C1_true_beats_orig"] = R("TRUE") - R("ORIG") >= 0.20
            crit["C2_true_beats_perm"] = R("TRUE") - R("PERM") >= 0.20
            crit["C3_true_beats_rand"] = R("TRUE") - R("RAND") >= 0.20
            crit["C4_true_beats_conventional"] = R("TRUE") - R("CONV") >= 0.20
            crit["C5_abstention_equals_original"] = all(tab[v]["identical_tree_to_orig"] == len(seeds) for v in ("EMPTY", "INCONSISTENT", "BADMATCH"))
            crit["C6_ambiguous_no_spurious_gain"] = abs(R("AMBIGUOUS") - R("ORIG")) <= 0.10
            crit["C7_no_spurious_advantage"] = all(R(v) - R("ORIG") <= 0.10 for v in ("IRRELEVANT", "INCOMPLETE", "WRONG_SPLIT"))
            sem = [v for v in tab if v not in ("ORIG", "WIDE", "CONV")]
            crit["C8_bounded_harm"] = all(R(v) - R("ORIG") >= -0.20 and tab[v]["fidelity_mean"] >= tab["ORIG"]["fidelity_mean"] - 0.02 for v in sem)
            crit["C9_noisy_superset_reported"] = {"recovery": R("NOISY_SUPERSET"), "delta_vs_orig": R("NOISY_SUPERSET") - R("ORIG")}
        else:
            crit["D1_true_flat_no_advantage"] = R("TRUE") - R("ORIG") <= 0.10
            crit["D2_hierarchy_advantage"] = R("HIERARCHY") - R("ORIG") >= 0.20
            crit["D3_perm_no_advantage"] = R("PERM") - R("ORIG") <= 0.10
        out[pname]["criteria"] = crit
    return out


def to_markdown(an):
    L = []
    for pname, d in an.items():
        L.append(f"### {pname} (n = {d['n_seeds']} sementes de avaliação)\n")
        L.append("| variante | recuperação [IC95 Wilson] | Δ vs ORIG | McNemar (b,c; p) | fidelidade teste (média ± DP) | Δ fid vs ORIG | decisões alteradas/árvore | árvores idênticas a ORIG | tempo/ORIG |\n|---|---|---|---|---|---|---|---|---|")
        for v, t in d["table"].items():
            ci = t["recovery_ci95"]
            dv = t.get("delta_recovery_vs_orig")
            dfid = t.get("delta_fidelity_vs_orig")
            bc = t.get("mcnemar_b_c")
            dv_s = "" if dv is None else "%+.3f" % dv
            mc_s = "" if dv is None else "(%d,%d); %.3f" % (bc[0], bc[1], t["mcnemar_p"])
            df_s = "" if dfid is None else "%+.4f" % dfid
            L.append("| %s | %.3f [%.2f; %.2f] | %s | %s | %.4f ± %.4f | %s | %.2f | %d/%d | %.2f |" % (
                v, t["recovery"], ci[0], ci[1], dv_s, mc_s, t["fidelity_mean"], t["fidelity_sd"], df_s, t["choice_changed_mean"],
                t["identical_tree_to_orig"], d["n_seeds"], t["time_ratio_vs_orig"]))
        L.append("\nCritérios pré-registados:\n")
        for k, v in d["criteria"].items():
            L.append(f"- {k}: **{v if isinstance(v, dict) else ('CUMPRIDO' if v else 'NÃO CUMPRIDO')}**")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    an = analyse(json.loads(Path(a.results).read_text()))
    Path(a.out).write_text(json.dumps(an, indent=1))
    Path(a.out).with_suffix(".md").write_text(to_markdown(an))
    print(to_markdown(an))


if __name__ == "__main__":
    main()
