"""Análise do fatorial S×A×L e dos controlos P2 (Doc. 23; protocolo pré-registado em docs/b1b4/23_ANALISE_CAUSAL_GANHO_COERENCIA.md §0).
Exploratória, ``observed_only``; só leitura de brutos; sem seleção. Células nativas r=0 vêm do benchmark da amostragem coerente (bit-a-bit reutilizáveis).
    python -m validation.benchmark.causal_analysis --cx DIR --coh DIR [--out FILE]
"""
from __future__ import annotations

import argparse
import json
import re
import statistics as st
from pathlib import Path

import numpy as np

from validation.benchmark.diag_b2_analysis import load

NATIVE_R0 = {("h", "on", "n"): "reloaded_owl_full", ("h", "off", "n"): "hist_active_off", ("c", "on", "n"): "coh_full", ("c", "off", "n"): "coh_active_off"}
CELL = re.compile(r"^cx_([hc])_(on|off)_([nf])_r(\d+)$")
P2 = re.compile(r"^p2_(stru|feat)shuf_(on|off)_r(\d+)$")
METRICS = ("fidelity_to_oracle", "accuracy_real_labels", "node_count", "depth", "fidelity_train", "oracle_queries")
HOLM = lambda ps: [min(1.0, p * (len(ps) - i)) for i, p in sorted(enumerate(ps), key=lambda t: t[1])]


def se(x):
    return st.stdev(x) / len(x) ** 0.5 if len(x) > 1 else None


def holm(pvals):
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, run = [None] * len(pvals), 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, pvals[i] * (len(pvals) - rank)))
        adj[i] = run
    return adj


def tpval(x):
    from scipy.stats import t as T
    n = len(x)
    if n < 2 or st.stdev(x) == 0:
        return 1.0
    return float(2 * T.sf(abs(st.mean(x) / (st.stdev(x) / n ** 0.5)), n - 1))


def build_cells(cx, coh, metric):
    """cells[unit][(S,A,L)] = [valores por réplica]; p2[unit][(mode,A)] = [valores]"""
    cells, p2, rep0 = {}, {}, {}
    for unit, arms in cx.items():
        c, q = {}, {}
        for aid, row in arms.items():
            m = CELL.match(aid)
            if m:
                c.setdefault((m.group(1), m.group(2), m.group(3)), {})[int(m.group(4))] = row[metric]
            m = P2.match(aid)
            if m:
                q.setdefault((m.group(1), m.group(2)), {})[int(m.group(3))] = row[metric]
        for key, aid in NATIVE_R0.items():
            if unit in coh and aid in coh[unit]:
                c.setdefault(key, {})[0] = coh[unit][aid][metric]
        cells[unit], p2[unit] = c, q
    return cells, p2


def cell_mean(d):
    return st.mean(d.values())


def factorial(cells, units, reps_limit=None):
    out = {}
    eff = {"S": [], "A": [], "L": [], "SxA": [], "SxL": [], "AxL": [], "SxAxL": []}
    for u in units:
        c = cells[u]
        if not all((S, A, L) in c for S in "hc" for A in ("on", "off") for L in "nf"):
            continue
        m = {k: cell_mean({r: v for r, v in c[k].items() if reps_limit is None or r < reps_limit}) for k in c}
        g = lambda S, A, L: m[(S, A, L)]
        sign = lambda S, A, L: ((1 if S == "c" else -1), (1 if A == "on" else -1), (1 if L == "f" else -1))
        vals = {k: 0.0 for k in eff}
        for S in "hc":
            for A in ("on", "off"):
                for L in "nf":
                    s, a, l = sign(S, A, L)
                    v = g(S, A, L) / 4.0
                    vals["S"] += s * v; vals["A"] += a * v; vals["L"] += l * v
                    vals["SxA"] += s * a * v; vals["SxL"] += s * l * v; vals["AxL"] += a * l * v; vals["SxAxL"] += s * a * l * v
        for k in eff:
            eff[k].append((u, vals[k]))
    return eff


def summarize_effects(eff):
    keys = list(eff)
    ps = [tpval([v for _, v in eff[k]]) for k in keys]
    adj = holm(ps)
    res = {}
    for k, p, pa in zip(keys, ps, adj):
        x = [v for _, v in eff[k]]
        by = {}
        for u, v in eff[k]:
            by.setdefault(u.split(":")[0], []).append(v)
        res[k] = {"mean": st.mean(x) if x else None, "se": se(x), "n_units": len(x), "p_uncorrected": p, "p_holm": pa, "worse": sum(v < -1e-9 for v in x),
                  "better": sum(v > 1e-9 for v in x), "macro_by_dataset": st.mean(st.mean(v) for v in by.values()) if by else None,
                  "per_dataset": {d: st.mean(v) for d, v in by.items()}}
    return res


def boot_ratio(num, den, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    num, den = np.asarray(num), np.asarray(den)
    r = []
    for _ in range(n):
        i = rng.integers(0, len(num), len(num))
        d = den[i].mean()
        if abs(d) > 1e-12:
            r.append(num[i].mean() / d)
    return {"point": float(num.mean() / den.mean()) if abs(den.mean()) > 1e-12 else None, "ci95": [float(np.quantile(r, 0.025)), float(np.quantile(r, 0.975))] if r else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cx", required=True); ap.add_argument("--coh", required=True); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cx_raw, coh_raw = load(Path(a.cx)), load(Path(a.coh))
    cx = {f"{k[0]}:{k[1]}": v for k, v in cx_raw.items()}
    coh = {f"{k[0]}:{k[1]}": v for k, v in coh_raw.items()}
    units = sorted(cx)
    res = {"units": units, "n_units": len(units)}
    # integridade: referências comuns idênticas ao benchmark anterior
    ref_ok = [all(cx[u][arm][m] == coh[u][arm][m] for m in ("fidelity_to_oracle", "accuracy_real_labels", "node_count") ) for u in units for arm in ("trepan_original", "reloaded_core", "c45") if u in coh and arm in cx[u] and arm in coh[u]]
    res["integrity_reference_arms_identical"] = {"compared": len(ref_ok), "identical": int(sum(ref_ok))}
    for metric in ("fidelity_to_oracle", "accuracy_real_labels", "node_count"):
        cells, p2 = build_cells(cx, coh, metric)
        eff = factorial(cells, units)
        res[f"factorial__{metric}"] = summarize_effects(eff)
        if metric == "fidelity_to_oracle":
            eff0 = factorial(cells, units, reps_limit=1)
            res["factorial_r0_only__fidelity"] = summarize_effects(eff0)
            # decomposição de D − C
            DC, dLc, dLh, dSf, dSn = [], [], [], [], []
            DCoff = []
            for u in units:
                c = cells[u]
                if not all((S, "on", L) in c for S in "hc" for L in "nf"):
                    continue
                m = {k: cell_mean(v) for k, v in c.items()}
                DC.append(m[("c", "on", "n")] - m[("h", "on", "n")])
                dLc.append(m[("c", "on", "n")] - m[("c", "on", "f")])            # se L fosse o mecanismo: coerente com família enriquecida perde o ganho
                dLh.append(m[("h", "on", "f")] - m[("h", "on", "n")])            # ou: histórico com família original ganha
                dSf.append(m[("c", "on", "f")] - m[("h", "on", "f")])
                dSn.append(m[("c", "on", "n")] - m[("h", "on", "n")])
            res["decomposition_D_minus_C"] = {"D_minus_C": {"mean": st.mean(DC), "se": se(DC)},
                                              "S_effect_at_L_native(=D−C)": {"mean": st.mean(dSn), "se": se(dSn)},
                                              "S_effect_at_L_flipped": {"mean": st.mean(dSf), "se": se(dSf)},
                                              "L_effect_in_coherent (native − flipped)": {"mean": st.mean(dLc), "se": se(dLc), "share_of_D_minus_C": boot_ratio(dLc, DC)},
                                              "L_effect_in_historical (flipped − native)": {"mean": st.mean(dLh), "se": se(dLh), "share_of_D_minus_C": boot_ratio(dLh, DC)},
                                              "criterion": "mecanismo L identificado se share ≥ 0,70 e |efeito| > 2 EP (pré-registo §0.5)"}
            # variância entre réplicas (ruído RNG): DP dentro da célula entre réplicas, por unidade
            sds = {}
            for u in units:
                for k, d in cells[u].items():
                    if len(d) >= 2:
                        sds.setdefault(k, []).append(st.pstdev(d.values()))
            res["replicate_sd_within_cell"] = {"/".join(k): st.mean(v) for k, v in sds.items()}
            res["replicate_sd_pooled"] = st.mean(x for v in sds.values() for x in v)
            # P2 vs coerente nativo (mesmas réplicas)
            P2c = {}
            for mode in ("stru", "feat"):
                for A in ("on", "off"):
                    diffs = []
                    for u in units:
                        q = p2[u].get((mode, A))
                        base = cells[u].get(("c", A, "n"))
                        if not q or not base:
                            continue
                        rs = sorted(set(q) & set(base))
                        if rs:
                            diffs.append(st.mean(base[r] for r in rs) - st.mean(q[r] for r in rs))     # verdadeiro − controlo parcial
                    P2c[f"true_minus_{mode}shuf_A={A}"] = {"mean": st.mean(diffs) if diffs else None, "se": se(diffs) if diffs else None, "n": len(diffs),
                                                           "worse": sum(v < -1e-9 for v in diffs), "better": sum(v > 1e-9 for v in diffs)}
            res["p2_true_minus_partial_permutation__fidelity"] = P2c
    cells_acc, p2_acc = build_cells(cx, coh, "accuracy_real_labels")
    P2a = {}
    for mode in ("stru", "feat"):
        for A in ("on", "off"):
            diffs = []
            for u in units:
                q = p2_acc[u].get((mode, A)); base = cells_acc[u].get(("c", A, "n"))
                if q and base:
                    rs = sorted(set(q) & set(base))
                    if rs:
                        diffs.append(st.mean(base[r] for r in rs) - st.mean(q[r] for r in rs))
            P2a[f"true_minus_{mode}shuf_A={A}"] = {"mean": st.mean(diffs) if diffs else None, "se": se(diffs) if diffs else None, "n": len(diffs)}
    res["p2_true_minus_partial_permutation__accuracy"] = P2a
    # indicadores de mecanismo (auditorias): adoção local, m-of-n, derivadas, consultas, oráculo
    aud = {}
    for f in sorted((Path(a.cx) / "audits").glob("*.json")):
        r = json.loads(f.read_text())
        m = CELL.match(r["arm"])
        key = "/".join(m.groups()[:3]) if m else r["arm"]
        e = aud.setdefault(key, {"adopt_rate": [], "mofn_share": [], "derived_share": [], "ks": [], "oor": [], "label_tv": [], "label_entropy": [], "accepted": [], "attempted": [], "mean_literals": []})
        if r.get("local_model_calls"):
            e["adopt_rate"].append(r["local_model_adopted"] / r["local_model_calls"])
        t = r.get("tree") or {}
        if t.get("n_splits"):
            e["mofn_share"].append(t["mofn_splits"] / t["n_splits"]); e["mean_literals"].append(t["mean_literals"])
            e["derived_share"].append(t["derived_literals"] / max(1, t["total_literals"]))
        q = r.get("queries") or {}
        for k2, k in (("ks_original_mean", "ks"), ("out_of_range_rate", "oor"), ("label_tv_vs_real", "label_tv"), ("label_entropy", "label_entropy")):
            if k2 in q:
                e[k].append(q[k2])
        s = r.get("summary") or {}
        if s:
            e["attempted"].append(s.get("error_focused_interventions_attempted", 0)); e["accepted"].append(s.get("error_focused_interventions_accepted", 0))
    res["mechanism_indicators"] = {k: {kk: (st.mean(v) if v and kk not in ("accepted", "attempted") else (sum(v) if v else None)) for kk, v in d.items()} for k, d in aud.items()}
    out = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(out)
    print(out)


if __name__ == "__main__":
    main()
