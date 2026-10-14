"""Análise do piloto DEV de B2-C (exploratória; observed_only; sem seleção por teste). Só leitura dos resultados brutos.
    python -m validation.benchmark.pilot_b2c_analysis --dir DIR [--out FILE.json]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from validation.benchmark.diag_b2_analysis import load

CONFIGS = ("legacy", "tiebreak", "calibrated_l0p2", "calibrated_l0p4")
NOB2C = {"legacy": "reloaded_owl_full", "tiebreak": "b1_tiebreak", "calibrated_l0p2": "b1_calibrated_l0p2", "calibrated_l0p4": "b1_calibrated_l0p4"}


def _pair(units, a, b, m):
    return [u[a][m] - u[b][m] for u in units.values() if a in u and b in u and u[a].get(m) is not None and u[b].get(m) is not None]


def _stat(x):
    n = len(x)
    if not n:
        return None
    sd = st.stdev(x) if n > 1 else 0.0
    return {"n": n, "mean": st.mean(x), "se": sd / n ** 0.5 if n > 1 else None, "sd": sd, "worse": sum(v < -1e-9 for v in x), "better": sum(v > 1e-9 for v in x),
            "equal": sum(abs(v) <= 1e-9 for v in x)}


def _macro(units, a, b, m):
    by = {}
    for (ds, _s), u in units.items():
        if a in u and b in u:
            by.setdefault(ds, []).append(u[a][m] - u[b][m])
    per = {d: st.mean(v) for d, v in by.items()}
    return (st.mean(per.values()) if per else None), per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.dir)
    U = load(root)
    res = {"units": sorted(f"{k[0]}:{k[1]}" for k in U), "contrasts": {}, "hierarchy": {}, "b2c": {}, "cost": {}}
    arms = ["trepan_original", "trepan_original_enr", "reloaded_owl_full", "reloaded_owl_shuffled"] + [NOB2C[c] for c in CONFIGS[1:]] + \
        [f"b2c_{c}{s}" for c in CONFIGS for s in ("", "_shuf")]
    for arm in arms:
        ent = {}
        for ref in ("trepan_original", "trepan_original_enr"):
            for m, key in (("fidelity_to_oracle", "dF"), ("accuracy_real_labels", "dAcc"), ("node_count", "dNodes"), ("depth", "dDepth")):
                ent[f"{key}_vs_{ref}"] = {"macro": _macro(U, arm, ref, m)[0], "per_dataset": _macro(U, arm, ref, m)[1], "paired": _stat(_pair(U, arm, ref, m))}
        res["contrasts"][arm] = ent
    for c in CONFIGS:
        b, n, s = f"b2c_{c}", NOB2C[c], f"b2c_{c}_shuf"
        ent = {}
        for m, key in (("fidelity_to_oracle", "dF"), ("accuracy_real_labels", "dAcc"), ("node_count", "dNodes")):
            ent[f"{key}_b2c_minus_noB2C"] = {"macro": _macro(U, b, n, m)[0], "paired": _stat(_pair(U, b, n, m)), "per_dataset": _macro(U, b, n, m)[1]}
            ent[f"{key}_b2c_minus_permuted_b2c"] = {"macro": _macro(U, b, s, m)[0], "paired": _stat(_pair(U, b, s, m))}
        res["contrasts"][f"__{c}__"] = ent
        # hierarquia por unidade (Acc_R > Acc_O > Acc_C45; F_R > F_O), sobre a variante b2c
        H = {"acc_R_gt_O": 0, "acc_O_gt_C45": 0, "acc_full_order": 0, "fid_R_gt_O": 0, "fid_R_ge_O": 0, "units": 0}
        for u in U.values():
            if not all(k in u for k in (b, "trepan_original", "c45")):
                continue
            H["units"] += 1
            ar, ao, ac = (u[b]["accuracy_real_labels"], u["trepan_original"]["accuracy_real_labels"], u["c45"]["accuracy_real_labels"])
            fr, fo = u[b]["fidelity_to_oracle"], u["trepan_original"]["fidelity_to_oracle"]
            H["acc_R_gt_O"] += ar > ao; H["acc_O_gt_C45"] += ao > ac; H["acc_full_order"] += (ar > ao > ac)
            H["fid_R_gt_O"] += fr > fo; H["fid_R_ge_O"] += fr >= fo
        res["hierarchy"][b] = H
        # B2-C: auditoria agregada (linhas de resultado)
        rows = [u[b] for u in U.values() if b in u]
        res["b2c"][b] = {k: sum((r.get(k) or 0) for r in rows) for k in ("b2c_nodes", "b2c_accepted", "b2c_fallback_used", "b2c_conditions_met", "b2c_diverged", "b2c_semantic_queries")}
        res["b2c"][b]["accept_rate"] = (res["b2c"][b]["b2c_accepted"] / res["b2c"][b]["b2c_nodes"]) if res["b2c"][b]["b2c_nodes"] else None
        res["b2c"][b]["fallback_rate"] = (res["b2c"][b]["b2c_fallback_used"] / res["b2c"][b]["b2c_nodes"]) if res["b2c"][b]["b2c_nodes"] else None
        # custo: consultas e tempo de treino
        for nm in (b, n):
            ur = [(u[nm], u["trepan_original"]) for u in U.values() if nm in u and "trepan_original" in u]
            res["cost"][nm] = {"oracle_queries_ratio": st.mean([x["oracle_queries"] / max(1.0, o["oracle_queries"]) for x, o in ur]) if ur else None,
                               "train_time_ratio": st.mean([x["tree_training_time"] / max(1e-9, o["tree_training_time"]) for x, o in ur]) if ur else None,
                               "total_oracle_samples_mean": st.mean([x.get("total_oracle_samples") or 0 for x, _ in ur]) if ur else None}
    out = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
