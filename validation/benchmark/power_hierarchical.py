"""Potência/precisão para a estrutura HIERÁRQUICA dataset × semente (D6). Só leitura de resultados DEV; nenhum dado confirmatório.

Modelo: ΔF_{d,s} = μ + a_d + e_{d,s},  a_d ~ N(0, σ_d²) (heterogeneidade ENTRE datasets), e ~ N(0, σ_e²) (variabilidade entre sementes DENTRO do dataset).
Estimador: média dos ΔF sobre m datasets × s sementes ⇒ Var = σ_d²/m + σ_e²/(m·s). As sementes do mesmo dataset NÃO são independentes entre datasets:
só reduzem o termo σ_e². O n.º de datasets independentes (m) é o fator limitante quando σ_d² > 0.
    python -m validation.benchmark.power_hierarchical --dir DIR --arm ARM [--ref trepan_original] [--out FILE]
Componentes de variância por ANOVA de 1 via (estimador dos momentos; com ≤ 4 datasets a estimativa de σ_d² é MUITO imprecisa — reportado como tal).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

from validation.benchmark.diag_b2_analysis import load

Z = {0.05: 1.645, 0.025: 1.96}
ZB = {0.8: 0.8416, 0.9: 1.2816}


def components(units, arm, ref, metric="fidelity_to_oracle"):
    by = {}
    for (ds, _s), u in units.items():
        if arm in u and ref in u:
            by.setdefault(ds, []).append(u[arm][metric] - u[ref][metric])
    m = len(by)
    if m < 2:
        return None
    ns = [len(v) for v in by.values()]
    N = sum(ns)
    grand = sum(sum(v) for v in by.values()) / N
    ssw = sum(sum((x - st.mean(v)) ** 2 for x in v) for v in by.values())
    dfw = N - m
    msw = ssw / dfw if dfw > 0 else 0.0
    ssb = sum(len(v) * (st.mean(v) - grand) ** 2 for v in by.values())
    msb = ssb / (m - 1)
    n0 = (N - sum(n * n for n in ns) / N) / (m - 1)
    sig_d2 = max(0.0, (msb - msw) / n0)
    return {"datasets": m, "units": N, "mean": grand, "sigma_e": math.sqrt(msw), "sigma_d": math.sqrt(sig_d2), "per_dataset_mean": {d: st.mean(v) for d, v in by.items()}}


def required_datasets(sig_d, sig_e, s, delta, alpha=0.05, power=0.8):
    var1 = sig_d ** 2 + sig_e ** 2 / s
    return math.ceil(((Z[alpha] + ZB[power]) ** 2) * var1 / delta ** 2)


def scenarios(sig_d, sig_e, delta=0.01):
    out = {}
    for name, (s_list) in {"minimo": (5,), "recomendado": (10,), "alargado": (20,)}.items():
        for s in s_list:
            out[f"{name}_s{s}"] = {"seeds_per_dataset": s, "datasets_needed_alpha0.05_power0.8": required_datasets(sig_d, sig_e, s, delta),
                                   "datasets_needed_alpha0.05_power0.9": required_datasets(sig_d, sig_e, s, delta, power=0.9),
                                   "executions_power0.8": required_datasets(sig_d, sig_e, s, delta) * s}
    # limite quando s→∞: só σ_d conta
    out["limite_seeds_infinitas"] = {"datasets_needed_alpha0.05_power0.8": math.ceil(((Z[0.05] + ZB[0.8]) ** 2) * sig_d ** 2 / delta ** 2) if sig_d > 0 else 0}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True); ap.add_argument("--arm", required=True); ap.add_argument("--ref", default="trepan_original")
    ap.add_argument("--delta", type=float, default=0.01); ap.add_argument("--out", default=None)
    a = ap.parse_args()
    c = components(load(Path(a.dir)), a.arm, a.ref)
    res = {"arm": a.arm, "ref": a.ref, "delta": a.delta, "components": c, "scenarios": scenarios(c["sigma_d"], c["sigma_e"], a.delta) if c else None,
           "warning": "σ_d estimado com poucos datasets (DEV: 4) é muito instável; usar apenas como ordem de grandeza e para o desenho, não como inferência"}
    s = json.dumps(res, indent=1, default=str)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
