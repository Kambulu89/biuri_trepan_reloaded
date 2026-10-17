"""Análise da avaliação emparelhada do amostrador C (Doc. 26 §0.3; EXPLORATÓRIA, DEV, ``observed_only``; só leitura de brutos).
    python -m validation.benchmark.sampler_c_analysis --sc DIR [--cx DIR_DOC23] [--coh DIR] [--out FILE.json]
Réplicas: A,C,E r=0 vêm do benchmark da própria unidade; r>0 de A,E = ``sc_A/E_r``; de C = células ``cx_h_on_n_r`` (Doc. 23; digits r=2 = ``sc_C_r2``).
Cada unidade (dataset, semente mestra) usa a MÉDIA das réplicas; EP entre unidades; réplicas e unidades do mesmo dataset NÃO são independentes.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

import numpy as np

from validation.benchmark.causal_analysis import holm, se, tpval
from validation.benchmark.diag_b2_analysis import load

METRICS = ("fidelity_to_oracle", "accuracy_real_labels", "node_count", "depth", "oracle_queries", "tree_training_time")
EQ_KEYS = ("fidelity_to_oracle", "accuracy_real_labels", "node_count", "depth", "oracle_queries")


def replicas(arms: dict, cx_arms: dict, family: str, metric: str, reps: int = 3) -> dict:
    """{r: valor} para a família A..F de uma unidade."""
    out = {}
    for r in range(reps):
        if family == "A":
            row = arms.get("trepan_original") if r == 0 else arms.get(f"sc_A_r{r}")
        elif family == "E":
            row = arms.get("reloaded_core") if r == 0 else arms.get(f"sc_E_r{r}")
        elif family == "C":
            row = arms.get("reloaded_owl_full") if r == 0 else (arms.get(f"sc_C_r{r}") or (cx_arms or {}).get(f"cx_h_on_n_r{r}"))
        else:
            row = arms.get(f"sc_{family}_r{r}")
        if row is not None and row.get(metric) is not None:
            out[r] = float(row[metric])
    return out


def unit_mean(arms, cx_arms, family, metric):
    v = replicas(arms, cx_arms, family, metric)
    return float(np.mean(list(v.values()))) if v else None


def contrast(units, cx, fa, fb, metric):
    d = []
    for u, arms in units.items():
        a, b = unit_mean(arms, cx.get(u), fa, metric), unit_mean(arms, cx.get(u), fb, metric)
        if a is not None and b is not None:
            d.append((u, a - b))
    return d


def summarize(d):
    x = [v for _, v in d]
    ds = {}
    for u, v in d:
        ds.setdefault(u.split(":")[0], []).append(v)
    per_ds = {k: float(np.mean(v)) for k, v in ds.items()}
    s = se(x)
    return {"mean": float(np.mean(x)), "se": s, "n_units": len(x), "p": tpval(x), "better": int(sum(v > 1e-12 for v in x)), "worse": int(sum(v < -1e-12 for v in x)),
            "macro_by_dataset": float(np.mean(list(per_ds.values()))), "per_dataset": per_ds, "ratio_to_se": (float(np.mean(x)) / s) if s else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True)
    ap.add_argument("--cx")
    ap.add_argument("--out")
    a = ap.parse_args()
    sc_raw = load(Path(a.sc))
    units = {f"{k[0]}:{k[1]}": v for k, v in sc_raw.items()}
    cx = {f"{k[0]}:{k[1]}": v for k, v in load(Path(a.cx)).items()} if a.cx else {}
    res = {"units": sorted(units), "n_units": len(units)}
    # EQ
    eq = {}
    for name, (f1, f2) in {"EQ_hist_A_vs_E": ("A", "E"), "EQ_C_B_vs_F": ("B", "F")}.items():
        n = same = 0
        diffs = []
        for u, arms in units.items():
            for r in range(3):
                r1 = replicas(arms, cx.get(u), f1, "fidelity_to_oracle")
                if r not in r1:
                    continue
                ok = True
                for m in EQ_KEYS:
                    x, y = replicas(arms, cx.get(u), f1, m).get(r), replicas(arms, cx.get(u), f2, m).get(r)
                    if x is None or y is None:
                        continue
                    ok &= (x == y)
                n += 1
                same += int(ok)
                if not ok:
                    diffs.append(f"{u}:r{r}")
        eq[name] = {"compared": n, "identical": same, "different": diffs[:20]}
    res["equivalence"] = eq
    # descritiva por família
    desc = {}
    for fam in "ABCDEF":
        desc[fam] = {}
        for m in METRICS:
            vals = [v for v in (unit_mean(arms, cx.get(u), fam, m) for u, arms in units.items()) if v is not None]
            desc[fam][m] = float(np.mean(vals)) if vals else None
    c45 = [arms["c45"] for arms in units.values() if "c45" in arms]
    desc["C45"] = {m: float(np.mean([r[m] for r in c45 if r.get(m) is not None])) for m in ("fidelity_to_oracle", "accuracy_real_labels", "node_count", "depth") if c45}
    res["descriptive_unit_means"] = desc
    # contrastes
    spec = {"B-A": ("B", "A"), "D-B": ("D", "B"), "D-C": ("D", "C"), "C-A": ("C", "A"), "(D-B)-(C-A)": None, "D-F": ("D", "F"), "E-A": ("E", "A"), "F-B": ("F", "B")}
    for metric, tag in (("fidelity_to_oracle", "fidelity"), ("accuracy_real_labels", "accuracy"), ("node_count", "nodes"), ("depth", "depth"), ("oracle_queries", "queries"),
                        ("tree_training_time", "train_time")):
        block = {}
        for name, ab in spec.items():
            if ab is None:
                db, ca = dict(contrast(units, cx, "D", "B", metric)), dict(contrast(units, cx, "C", "A", metric))
                d = [(u, db[u] - ca[u]) for u in db if u in ca]
            else:
                d = contrast(units, cx, ab[0], ab[1], metric)
            if d:
                block[name] = summarize(d)
        fam = ["B-A", "D-B", "D-C", "C-A", "(D-B)-(C-A)"]
        if all(f in block for f in fam) and tag in ("fidelity", "accuracy"):
            adj = holm([block[f]["p"] for f in fam])
            for f, p in zip(fam, adj):
                block[f]["p_holm_family5"] = p
        res[f"contrasts__{tag}"] = block
    # ruído de réplica (desvio-padrão entre réplicas dentro da unidade)
    sd = {}
    for fam in "ABCDEF":
        v = [np.std(list(replicas(arms, cx.get(u), fam, "fidelity_to_oracle").values()), ddof=1) for u, arms in units.items() if len(replicas(arms, cx.get(u), fam, "fidelity_to_oracle")) > 1]
        sd[fam] = float(np.mean(v)) if v else None
    res["replicate_sd_fidelity_within_unit"] = sd
    # critérios pré-registados
    def ok(name, tag, positive=True, k=2.0):
        c = res[f"contrasts__{tag}"].get(name)
        return None if c is None else bool(c["se"] and c["mean"] > k * c["se"])
    cF, cA = res["contrasts__fidelity"], res["contrasts__accuracy"]
    res["criteria"] = {
        "C_improves_Original(B-A: dF>2SE and dAcc>=0)": bool(ok("B-A", "fidelity") and cA["B-A"]["mean"] >= 0),
        "ontological_gain_under_C(D-B: dF>2SE and dAcc>2SE)": bool(ok("D-B", "fidelity") and ok("D-B", "accuracy")),
    }
    # hierarquia por unidade (médias das réplicas)
    h = {"acc_D_gt_B": 0, "acc_B_gt_C45": 0, "fid_D_gt_B": 0, "all_three": 0, "n": 0}
    for u, arms in units.items():
        D, B = unit_mean(arms, cx.get(u), "D", "accuracy_real_labels"), unit_mean(arms, cx.get(u), "B", "accuracy_real_labels")
        FD, FB = unit_mean(arms, cx.get(u), "D", "fidelity_to_oracle"), unit_mean(arms, cx.get(u), "B", "fidelity_to_oracle")
        c = arms.get("c45", {}).get("accuracy_real_labels")
        if None in (D, B, FD, FB, c):
            continue
        h["n"] += 1
        h["acc_D_gt_B"] += int(D > B + 1e-12)
        h["acc_B_gt_C45"] += int(B > c + 1e-12)
        h["fid_D_gt_B"] += int(FD > FB + 1e-12)
        h["all_three"] += int(D > B + 1e-12 and B > c + 1e-12 and FD > FB + 1e-12)
    h["status"] = ("HIERARCHY_SATISFIED" if h["n"] and h["all_three"] == h["n"] and res["criteria"]["ontological_gain_under_C(D-B: dF>2SE and dAcc>2SE)"]
                   else "NOT_SATISFIED" if h["n"] and (h["acc_D_gt_B"] <= h["n"] // 2 or h["fid_D_gt_B"] <= h["n"] // 2)
                   else "INSUFFICIENT_EVIDENCE")
    res["hierarchy_counts"] = h
    # plausibilidade (auditorias das consultas efetivas)
    pl = {}
    for f in sorted((Path(a.sc) / "audits").glob("*.json")):
        rec = json.loads(f.read_text())
        arm = rec["arm"]
        fam = ("A" if arm == "trepan_original" else "E" if arm == "reloaded_core" else "C" if arm == "reloaded_owl_full" else arm.split("_")[1] if arm.startswith("sc_") else None)
        if fam is None:
            continue
        for k, v in rec["plausibility"].items():
            if isinstance(v, (int, float)):
                pl.setdefault(fam, {}).setdefault(k, []).append(float(v))
    res["plausibility_effective_queries"] = {fam: {k: float(np.mean(v)) for k, v in d.items()} for fam, d in pl.items()}
    # custo
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: v for k, v in res.items() if k in ("n_units", "equivalence", "criteria", "hierarchy_counts")}, indent=1))


if __name__ == "__main__":
    main()
