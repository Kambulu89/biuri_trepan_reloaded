"""B3.3: análise pré-registada (critérios em ``b33_run.CRITERIA33``). Reconstrói tudo só dos resultados brutos."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

CONTROLS_ONTO = ("D_PERM", "E_RAND", "I_NEUTRALIZED")
CONTROLS_GENERIC = ("B_WIDE", "F_CONV", "G_UNI_SCREEN", "H_PAIR_SCREEN", "J1_WIDE_BEAM_CAND", "J2_WIDE_BEAM_CPU")
R1_ARMS = ("D_PERM", "E_RAND", "F_CONV", "G_UNI_SCREEN", "H_PAIR_SCREEN", "I_NEUTRALIZED")
METRICS = ("recovered_path", "pair_same_test", "accuracy", "fidelity", "precision_macro", "macro_f1", "queries", "eval_calls", "literals_listed", "simple_generated", "mofn_generated",
           "oracle_samples", "internal_nodes", "depth", "mofn_nodes", "mean_rule_literals", "cpu_fit_s", "compile_wall_s", "screen_wall_s", "total_cost_cpu_s", "n_units", "unit_searches")


def _by_arm(rows: List[Dict[str, Any]]) -> Dict[str, Dict[int, Dict[str, Any]]]:
    out: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r in rows:
        out.setdefault(r["arm"], {})[r["seed"]] = r
    return out


def _diff(a: Dict[int, Dict], b: Dict[int, Dict], key: str, rng, n_boot=4000):
    seeds = sorted(set(a) & set(b))
    d = np.array([float(a[s][key]) - float(b[s][key]) for s in seeds])
    if len(d) == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    bs = rng.choice(d, size=(n_boot, len(d)), replace=True).mean(axis=1)
    return {"mean": float(d.mean()), "lo": float(np.quantile(bs, 0.025)), "hi": float(np.quantile(bs, 0.975)), "n": int(len(d))}


def analyse_family(res: Dict[str, Any]) -> Dict[str, Any]:
    rng = np.random.default_rng(33033)
    arms = _by_arm(res["rows"])
    mean = lambda arm, k: float(np.mean([r[k] for r in arms[arm].values()]))
    sd = lambda arm, k: float(np.std([r[k] for r in arms[arm].values()], ddof=1))
    table = {a: {k: {"mean": mean(a, k), "sd": sd(a, k)} for k in METRICS} for a in arms}
    refs = res["refs"]
    ref = {"c45_accuracy": float(np.mean([r["accuracy"] for r in refs])), "c45_fidelity": float(np.mean([r["fidelity"] for r in refs])), "mlp_accuracy": float(np.mean([r["mlp_accuracy"] for r in refs]))}
    rec = {a: table[a]["recovered_path"]["mean"] for a in table}
    r1_gap = rec["C_OWL"] - max(rec[a] for a in R1_ARMS)
    r1 = r1_gap >= 0.20
    r2 = (rec["C_OWL"] - rec["B_WIDE"] >= 0.10) and (rec["C_OWL"] - max(rec["J1_WIDE_BEAM_CAND"], rec["J2_WIDE_BEAM_CPU"]) >= 0.10)
    vsA = {k: _diff(arms["C_OWL"], arms["A_ORIG"], k, rng) for k in ("accuracy", "fidelity", "recovered_path")}
    if vsA["accuracy"]["mean"] > 0 and vsA["fidelity"]["mean"] > 0:
        joint = "SUPPORTED_JOINT_GAIN" if vsA["accuracy"]["lo"] > 0 and vsA["fidelity"]["lo"] > 0 else "OBSERVED_JOINT_GAIN"
    elif (vsA["accuracy"]["mean"] > 0) != (vsA["fidelity"]["mean"] > 0):
        joint = "TRADEOFF"
    else:
        joint = "NO_JOINT_GAIN"
    vs_ctrl, attributed = {}, joint == "SUPPORTED_JOINT_GAIN"
    for x in CONTROLS_ONTO + CONTROLS_GENERIC:
        da, df = _diff(arms["C_OWL"], arms[x], "accuracy", rng), _diff(arms["C_OWL"], arms[x], "fidelity", rng)
        vs_ctrl[x] = {"accuracy": da, "fidelity": df, "recovery_diff": _diff(arms["C_OWL"], arms[x], "recovered_path", rng)["mean"],
                      "joint_gain_over_control": bool(da["lo"] is not None and da["lo"] > 0 and df["lo"] > 0)}
        attributed = attributed and vs_ctrl[x]["joint_gain_over_control"]
    cheaper = {}
    for x in CONTROLS_GENERIC + CONTROLS_ONTO:
        eq = abs(vs_ctrl[x]["accuracy"]["mean"]) <= 0.005 and abs(vs_ctrl[x]["fidelity"]["mean"]) <= 0.005
        cost_x, cost_c = table[x]["total_cost_cpu_s"]["mean"], table["C_OWL"]["total_cost_cpu_s"]["mean"]
        cheaper[x] = bool((eq or (rec[x] >= rec["C_OWL"] and vs_ctrl[x]["accuracy"]["mean"] <= 0.005 and vs_ctrl[x]["fidelity"]["mean"] <= 0.005)) and cost_x <= 0.8 * cost_c)
    uninformative = max(rec[a] for a in rec if a not in ("N_IRRELEVANT", "N_INCOMPLETE")) < 0.10
    verdict = ("UNINFORMATIVE_NO_ARM_RECOVERS" if uninformative else "ONTOLOGICAL_ADVANTAGE" if (r1 and r2 and attributed) else "RELATION_SPECIFIC_ONLY" if r1 else "NO_ONTOLOGICAL_ADVANTAGE")
    hier = {"acc_C_gt_A_gt_C45": bool(table["C_OWL"]["accuracy"]["mean"] > table["A_ORIG"]["accuracy"]["mean"] > ref["c45_accuracy"]),
            "fid_C_gt_A": bool(table["C_OWL"]["fidelity"]["mean"] > table["A_ORIG"]["fidelity"]["mean"]), "fid_C_gt_C45": bool(table["C_OWL"]["fidelity"]["mean"] > ref["c45_fidelity"])}
    return {"family": res["family"], "n_seeds": len(arms["A_ORIG"]), "table": table, "references": ref, "recovery": rec, "R1_gap": r1_gap, "R1": r1, "R2": r2, "C_vs_A": vsA,
            "joint_gain_status": joint, "ontology_attribution": "ATTRIBUTED" if attributed else "NOT_ATTRIBUTED", "C_vs_controls": vs_ctrl, "SIMPLER_EQUIVALENT": cheaper,
            "verdict": verdict, "hierarchy_exploratory": hier}


def global_verdict(fams: Dict[str, Dict[str, Any]]) -> str:
    n = sum(1 for k, v in fams.items() if k[:2] in ("I3", "I4", "I5", "I6") and v["verdict"] == "ONTOLOGICAL_ADVANTAGE")
    return "B33_ADVANTAGE_ESTABLISHED" if n >= 2 else "B33_PARTIAL" if n == 1 else "B33_NO_ONTOLOGICAL_ADVANTAGE"


def main(paths: List[str], out_json: str) -> None:
    fams = {}
    for p in paths:
        res = json.loads(Path(p).read_text())
        fams[res["family"]] = analyse_family(res)
    doc = {"families": fams, "global_verdict": global_verdict(fams)}
    Path(out_json).write_text(json.dumps(doc, indent=1))


if __name__ == "__main__":
    import sys
    main(sys.argv[2:], sys.argv[1])
