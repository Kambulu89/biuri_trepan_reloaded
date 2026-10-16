"""B3.0 (Doc. 31 §3): auditoria SOMENTE LEITURA dos controlos G29/E29/Breast Cancer da campanha M29 a partir dos brutos. Não altera classificações."""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict
from pathlib import Path


def _row(p):
    return json.loads(Path(p).read_text())["row"]


def g29_units(sc_dir, m29_dirs):
    out = {}
    for ds_seed in sorted({tuple(Path(p).parts[-3:-1]) for d in m29_dirs for p in glob.glob(f"{d}/raw/*/seed*/m29_G_k0.json")}):
        ds, sd = ds_seed
        g = {}
        for d in m29_dirs:                                    # os últimos diretórios prevalecem (G29 do piloto repetido)
            for k in range(3):
                f = Path(d) / "raw" / ds / sd / f"m29_G_k{k}.json"
                if f.exists():
                    g[k] = _row(f)
        for k, r in sorted(g.items()):
            b = _row(Path(sc_dir) / "raw" / ds / sd / f"sc_D_r{k}.json")
            a = _row(Path(sc_dir) / "raw" / ds / sd / f"sc_B_r{k}.json")
            meta = json.loads(r["control_meta"])
            qg, qb, qa = float(r["oracle_queries"]), float(b["oracle_queries"]), float(a["oracle_queries"])
            out[(ds, sd, k)] = dict(requested=qb, executed=qg, diff=qg - qb, ratio=qg / qb, factor=meta.get("factor"), min_sample_used=meta.get("min_sample_used"),
                                    min_sample_canonical=meta.get("min_sample_canonical"), q_A29=qa, q_B29=qb,
                                    queries_per_node_G=qg / max(r["node_count"], 1), queries_per_node_B=qb / max(b["node_count"], 1),
                                    node_cap_G=bool(r["node_cap_reached"]), node_cap_B=bool(b.get("node_cap_reached")),
                                    nodes_G=r["node_count"], nodes_B=b["node_count"], valid=0.8 <= qg / qb <= 1.25)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sc", required=True); ap.add_argument("--m29", nargs="+", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    g = g29_units(a.sc, a.m29)
    per_unit = defaultdict(list)
    for (ds, sd, k), v in g.items():
        per_unit[f"{ds}:{sd[4:]}"].append(v)
    summ = {}
    for u, vs in per_unit.items():
        summ[u] = dict(n_draws=len(vs), requested_mean=sum(v["requested"] for v in vs) / len(vs), executed_mean=sum(v["executed"] for v in vs) / len(vs),
                       ratio_mean=sum(v["ratio"] for v in vs) / len(vs), ratio_min=min(v["ratio"] for v in vs), ratio_max=max(v["ratio"] for v in vs),
                       valid_draws=sum(v["valid"] for v in vs), qpn_G=sum(v["queries_per_node_G"] for v in vs) / len(vs), qpn_B=sum(v["queries_per_node_B"] for v in vs) / len(vs),
                       nodes_G=sum(v["nodes_G"] for v in vs) / len(vs), nodes_B=sum(v["nodes_B"] for v in vs) / len(vs),
                       node_cap_G=sum(v["node_cap_G"] for v in vs), node_cap_B=sum(v["node_cap_B"] for v in vs))
        summ[u]["unit_valid"] = summ[u]["valid_draws"] == len(vs) and 0.8 <= summ[u]["ratio_mean"] <= 1.25
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(dict(per_draw={f"{k[0]}:{k[1]}:r{k[2]}": v for k, v in g.items()}, per_unit=summ), indent=1))
    print("| unidade | pedido (B29) | executado (G29) | Δ | rácio (min–max) | draws válidos | q/nó G | q/nó B | nós G/B | teto de nós G/B | contraste válido |\n|---|---|---|---|---|---|---|---|---|---|---|")
    for u, s in sorted(summ.items()):
        print(f"| {u} | {s['requested_mean']:.0f} | {s['executed_mean']:.0f} | {s['executed_mean']-s['requested_mean']:+.0f} | {s['ratio_mean']:.2f} ({s['ratio_min']:.2f}–{s['ratio_max']:.2f}) | {s['valid_draws']}/{s['n_draws']} | {s['qpn_G']:.0f} | {s['qpn_B']:.0f} | {s['nodes_G']:.0f}/{s['nodes_B']:.0f} | {s['node_cap_G']}/{s['node_cap_B']} | {'sim' if s['unit_valid'] else 'NÃO'} |")


if __name__ == "__main__":
    main()


def reconcile(report_path):
    """Reconciliação (Doc. 31 §3): para cada unidade, razão do estado de atribuição e validade de cada controlo. NÃO altera classificações; só explica."""
    r = json.loads(Path(report_path).read_text())
    rows = []
    for u, pu in sorted(r["per_unit"].items()):
        st = pu["states"]
        cc = st.get("control_confounding") or {}
        var = pu.get("control_variants", {})
        ctl = {}
        for arm in ("C29", "D29", "E29", "G29"):
            c = cc.get(arm, {})
            v = var.get(arm, [])
            n_fb = sum(1 for x in v if x and x != "full")
            ctl[arm] = "fallback" if n_fb == len(v) and v else ("parcial" if n_fb else ("confundido" if c.get("confounded") else "válido"))
        if st["attribution"] == "NOT_ATTRIBUTED":
            motivo = "sem ganho conjunto direto (nada a atribuir)" if st["direct"] != "ONTOLOGICAL_JOINT_GAIN" else "ganho direto não se mantém sobre os controlos"
        else:
            motivo = "ganho direto mas controlo(s) exigido(s) confundido(s)/ausente(s): " + ",".join(st.get("missing") or [])
        rows.append(dict(unit=u, direct=st["direct"], attribution=st["attribution"], motivo=motivo, controls=ctl, global_state=st["global"]))
    return rows
