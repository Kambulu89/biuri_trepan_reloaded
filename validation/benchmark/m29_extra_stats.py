"""Estatísticas auxiliares do Doc. 30 (controlos, E29, G29, complexidade, regressões, tentativas de construção dos controlos). Só leitura."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from validation.benchmark.hierarchy_joint_report import load


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--m29", nargs="+", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sc = load(Path(a.sc))
    m = {}
    for d in a.m29:
        for k, arms in load(Path(d)).items():
            for arm_name, rec_ in arms.items():
                af = Path(d) / "audits" / f"{k[0]}_seed{k[1]}__{arm_name}.json"
                if af.exists():
                    meta = json.loads(af.read_text()).get("control") or {}
                    if "n_derived" in meta:
                        rec_["row"]["n_derived_effective"] = meta["n_derived"]
            m.setdefault(k, {}).update(arms)
    rep = json.loads(Path(a.report).read_text())
    res = {}
    # --- controlos P/D: variante, retenção, tentativas
    var = defaultdict(Counter)
    ret = defaultdict(list)
    fail_reasons = defaultdict(Counter)
    for (ds, seed), arms in m.items():
        for name, r in arms.items():
            if name.startswith(("m29_P", "m29_D")):
                var[ds][r["row"].get("control_variant")] += 1
                meta = json.loads(r["row"]["control_meta"])
                pm = meta.get("perm_meta") or {}
                if pm and meta["variant"] == "full":
                    ret[ds].append((pm.get("retention"), pm.get("chance_retention"), pm.get("met_criterion")))
                for at in meta["attempts"]:
                    for reason in at["reasons"]:
                        fail_reasons[ds][reason.split(" ")[0] + " " + (reason.split(" ")[1] if len(reason.split(" ")) > 1 else "")] += 1
    res["control_variants"] = {ds: dict(c) for ds, c in var.items()}
    res["retention_vs_chance_full_controls"] = {ds: {"mean_retention": float(np.mean([x[0] for x in v])), "mean_chance": float(np.mean([x[1] for x in v])), "all_met": all(x[2] for x in v), "n": len(v)} for ds, v in ret.items()}
    res["construction_failure_reasons"] = {ds: dict(c) for ds, c in fail_reasons.items()}
    # --- E29 e G29 vs B29
    e_info, g_info = defaultdict(list), defaultdict(list)
    for (ds, seed), arms in m.items():
        b = [sc[(ds, seed)][f"sc_D_r{k}"]["row"] for k in range(3)]
        for k in range(3):
            e = arms.get(f"m29_E_k{k}")
            if e:
                e_info[ds].append((b[k].get("semantic_features_generated"), e["row"].get("n_derived_effective"), e["row"]["oracle_queries"] / b[k]["oracle_queries"]))
            g = arms.get(f"m29_G_k{k}")
            if g:
                gm = json.loads(g["row"]["control_meta"])
                g_info[ds].append((gm["factor"], gm["ratio_to_B29"], gm["budget_matched"]))
    res["E29_vs_B29"] = {ds: {"derived_B29_mean": float(np.mean([x[0] for x in v])), "derived_E29_mean": float(np.mean([x[1] for x in v])), "queries_ratio_mean": float(np.mean([x[2] for x in v])), "n": len(v)} for ds, v in e_info.items()}
    res["G29_budget_match"] = {ds: {"factor_mean": float(np.mean([x[0] for x in v])), "ratio_to_B29_mean": float(np.mean([x[1] for x in v])), "fraction_matched": float(np.mean([x[2] for x in v])), "n": len(v)} for ds, v in g_info.items()}
    # --- complexidade e regressões por unidade (sobre as médias das unidades)
    nodes, regress, nb = [], Counter(), 0
    for u, d in rep["per_unit"].items():
        v = d["vals"]
        a_, b_ = v.get("A29"), v.get("B29")
        if not a_ or not b_:
            continue
        nb += 1
        nodes.append((u, b_["node_count"] / a_["node_count"]))
        regress["acc_down"] += int(b_["accuracy"] < a_["accuracy"])
        regress["fid_down"] += int(b_["fidelity"] < a_["fidelity"])
        regress["both_down"] += int(b_["accuracy"] < a_["accuracy"] and b_["fidelity"] < a_["fidelity"])
        regress["acc_up"] += int(b_["accuracy"] > a_["accuracy"])
        regress["fid_up"] += int(b_["fidelity"] > a_["fidelity"])
        regress["both_up"] += int(b_["accuracy"] > a_["accuracy"] and b_["fidelity"] > a_["fidelity"])
    res["B29_vs_A29_unit_regressions"] = {**regress, "n_units": nb}
    res["B29_nodes_ratio_vs_A29"] = {"max": max(r for _, r in nodes), "fraction_le_1_25": float(np.mean([r <= 1.25 for _, r in nodes])), "mean": float(np.mean([r for _, r in nodes]))}
    # --- unidades com ganho conjunto direto
    res["units_with_direct_joint_gain"] = {u: {"direct": d["states"]["direct"], "attribution": d["states"]["attribution"], "deltas_vs_A29": {k: d["vals"]["B29"][k] - d["vals"]["A29"][k] for k in ("accuracy", "fidelity")}}
                                           for u, d in rep["per_unit"].items() if d.get("states") and d["states"]["direct"] == "ONTOLOGICAL_JOINT_GAIN"}
    # --- razão de custo
    arms = rep["arms"]
    res["cost_ratios"] = {"queries_B29_over_A29": arms["B29"]["oracle_queries"] / arms["A29"]["oracle_queries"], "time_B29_over_A29": arms["B29"]["training_time"] / arms["A29"]["training_time"],
                          "queries_E29_over_B29": arms["E29"]["oracle_queries"] / arms["B29"]["oracle_queries"], "time_E29_over_A29": arms["E29"]["training_time"] / arms["A29"]["training_time"]}
    # --- E29 como ganho não ontológico (descritivo)
    res["E29_minus_A29_dataset_means"] = None
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
