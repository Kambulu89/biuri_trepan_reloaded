"""Análise do benchmark DEV da amostragem coerente (exploratória; observed_only; sem seleção por teste). Só leitura dos resultados brutos.
    python -m validation.benchmark.coherent_analysis --dir DIR [--out FILE.json]
Braços (Doc. 22): A trepan_original · B reloaded_core · C reloaded_owl_full (histórico, ativas) · D coh_full · E coh_active_off · F hist_active_off ·
G reloaded_owl_shuffled / coh_full_shuf / coh_active_off_shuf · H conv_coh · R trepan_original_enr · Rc coh_features.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

from validation.benchmark.diag_b2_analysis import load

A, B, C, D, E, F = "trepan_original", "reloaded_core", "reloaded_owl_full", "coh_full", "coh_active_off", "hist_active_off"
G1, G2, G3, H, R, RC = "reloaded_owl_shuffled", "coh_full_shuf", "coh_active_off_shuf", "conv_coh", "trepan_original_enr", "coh_features"
ALL = (A, B, C, D, E, F, G1, G2, G3, H, R, RC)
METRICS = (("fidelity_to_oracle", "dF"), ("accuracy_real_labels", "dAcc"), ("node_count", "dNodes"), ("depth", "dDepth"))


def stat(x):
    n = len(x)
    if not n:
        return None
    sd = st.stdev(x) if n > 1 else 0.0
    return {"n": n, "mean": st.mean(x), "se": sd / n ** 0.5 if n > 1 else None, "sd": sd, "worse": sum(v < -1e-9 for v in x), "better": sum(v > 1e-9 for v in x)}


def paired(U, a, b, m):
    by, allv = {}, []
    for (ds, _s), u in U.items():
        if a in u and b in u and u[a].get(m) is not None and u[b].get(m) is not None:
            v = u[a][m] - u[b][m]
            by.setdefault(ds, []).append(v)
            allv.append(v)
    per = {d: st.mean(v) for d, v in by.items()}
    return {"macro": st.mean(per.values()) if per else None, "per_dataset": per, "units": stat(allv)}


def contrast(U, a, b):
    return {key: paired(U, a, b, m) for m, key in METRICS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    root = Path(a.dir)
    U = load(root)
    res = {"units": sorted(f"{k[0]}:{k[1]}" for k in U), "vs_original": {}, "contrasts": {}, "hierarchy": {}, "cost": {}, "coherence": {}, "acceptance": {}}
    for arm in ALL[1:]:
        res["vs_original"][arm] = contrast(U, arm, A)
    named = {
        "coherence_effect_repr_only (Rc − R)": (RC, R),
        "coherence_effect_full_active (D − C)": (D, C),
        "coherence_effect_active_off (E − F)": (E, F),
        "active_queries_effect_historical (C − F)": (C, F),
        "active_queries_effect_coherent (D − E)": (D, E),
        "true_ontology_effect_coherent_active (D − G2)": (D, G2),
        "true_ontology_effect_coherent_active_off (E − G3)": (E, G3),
        "true_ontology_effect_historical (C − G1)": (C, G1),
        "owl_features_vs_conventional_repr_only (Rc − H)": (RC, H),
        "owl_full_coherent_vs_conventional (E − H)": (E, H),
        "reloaded_core_equals_original (B − A)": (B, A),
    }
    for k, (x, y) in named.items():
        res["contrasts"][k] = contrast(U, x, y)
    # interação coerência × consultas ativas = (D − C) − (E − F), emparelhada por unidade
    inter = {}
    for m, key in METRICS[:2]:
        v = [(u[D][m] - u[C][m]) - (u[E][m] - u[F][m]) for u in U.values() if all(k in u for k in (C, D, E, F))]
        inter[key] = stat(v)
    res["contrasts"]["interaction_coherence_x_active ((D−C)−(E−F))"] = inter
    for arm in (C, D, E, F, G1, G2, G3, H, R, RC):
        rows = [u[arm] for u in U.values() if arm in u]
        ref = [(u[arm], u[A]) for u in U.values() if arm in u and A in u]
        res["cost"][arm] = {"oracle_queries_ratio_vs_original": st.mean([x["oracle_queries"] / max(1.0, o["oracle_queries"]) for x, o in ref]) if ref else None,
                            "train_time_ratio_vs_original": st.mean([x["tree_training_time"] / max(1e-9, o["tree_training_time"]) for x, o in ref]) if ref else None,
                            "total_oracle_samples_mean": st.mean([r.get("total_oracle_samples") or 0 for r in rows]) if rows else None}
    for arm in (D, E):
        h = {"acc_R_gt_O": 0, "acc_O_gt_C45": 0, "acc_full_order": 0, "fid_R_gt_O": 0, "fid_R_ge_O": 0, "units": 0}
        for u in U.values():
            if all(k in u for k in (arm, A, "c45")):
                h["units"] += 1
                ar, ao, ac = u[arm]["accuracy_real_labels"], u[A]["accuracy_real_labels"], u["c45"]["accuracy_real_labels"]
                fr, fo = u[arm]["fidelity_to_oracle"], u[A]["fidelity_to_oracle"]
                h["acc_R_gt_O"] += ar > ao; h["acc_O_gt_C45"] += ao > ac; h["acc_full_order"] += (ar > ao > ac)
                h["fid_R_gt_O"] += fr > fo; h["fid_R_ge_O"] += fr >= fo
        res["hierarchy"][arm] = h
    # auditorias (coerência das consultas e aceitação semântica)
    coh = {}
    acc = {}
    for f in sorted((root / "audits").glob("*.json")):
        r = json.loads(f.read_text())
        arm = r["arm"]
        c = r.get("coherence") or {}
        for k, v in c.items():
            if isinstance(v, (int, float)) and v is not None:
                coh.setdefault(arm, {}).setdefault(k, []).append(float(v))
        s = r.get("summary") or {}
        if s:
            att, accd = s.get("error_focused_interventions_attempted", 0), s.get("error_focused_interventions_accepted", 0)
            e = acc.setdefault(arm, {"attempted": 0, "accepted": 0, "within_tolerance": 0, "evaluated": 0, "changed": 0})
            e["attempted"] += att; e["accepted"] += accd; e["within_tolerance"] += s.get("error_focused_accepted_within_tolerance", 0)
            e["evaluated"] += s.get("evaluated_splits", 0); e["changed"] += s.get("semantic_changed_splits", 0)
    res["coherence"] = {arm: {k: st.mean(v) for k, v in d.items()} for arm, d in coh.items()}
    res["acceptance"] = {arm: {**e, "accept_rate": (e["accepted"] / e["attempted"]) if e["attempted"] else None} for arm, e in acc.items()}
    out = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
