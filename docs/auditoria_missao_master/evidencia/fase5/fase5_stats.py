import json, glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
F5, F3U, F4U, OUTU, OUTC, OUTT, OUTSEL = sys.argv[1:8]
DS = ["iris", "wine", "breast_cancer"]
rows = []
for f in sorted(glob.glob(F5 + "/raw/*/seed*/sc_*.json")):
    b = os.path.basename(f)[:-5]; r = json.load(open(f))["row"]
    rows.append(dict(dataset=r["dataset"], seed=int(r["seed"]), kind=b.split("_")[1], acc=r["accuracy_real_labels"], fid=r["fidelity_to_oracle"],
                     nodes=r.get("node_count"), queries=r.get("membership_queries"), samples=r.get("total_oracle_samples") or r.get("membership_queries"),
                     secs=r.get("arm_time_s"), changed=r.get("semantic_decision_changed_count"), sem_splits=r.get("semantic_split_count")))
d = pd.DataFrame(rows)
print("unidades:", d.groupby("dataset").seed.nunique().to_dict(), "| réplicas:", d.groupby(["kind", "dataset", "seed"]).size().groupby("kind").agg(["min", "max"]).to_dict("index"))
u = d.groupby(["dataset", "seed", "kind"])[["acc", "fid", "nodes", "queries", "samples", "secs", "changed", "sem_splits"]].mean().reset_index()
f3 = pd.read_csv(F3U); f3 = f3[f3.dataset.isin(DS)]
f3 = f3[f3.cond.isin(["C4.5", "D2 Reloaded+OWL v2 + conjunto", "B Original + conjunto"])].assign(kind=lambda x: x.cond.map({"C4.5": "C4.5", "D2 Reloaded+OWL v2 + conjunto": "D2_f3", "B Original + conjunto": "B_f3"}))
f4 = pd.read_csv(F4U); f4 = f4[f4.kind == "O"].assign(kind="O_f4")
u = pd.concat([u, f3[["dataset", "seed", "kind", "acc", "fid"]], f4[["dataset", "seed", "kind", "acc", "fid"]]], ignore_index=True); u.to_csv(OUTU, index=False)
A = u.pivot_table(index=["dataset", "seed"], columns="kind", values="acc"); F = u.pivot_table(index=["dataset", "seed"], columns="kind", values="fid")
integ = {k: float(max((A[a] - A[b]).abs().max(), (F[a] - F[b]).abs().max())) for k, (a, b) in {"B vs fase 3": ("B", "B_f3"), "O vs fase 4": ("O", "O_f4"), "D vs D2 fase 3": ("D", "D2_f3")}.items()}
print("INTEGRIDADE:", integ)
m = u.groupby(["dataset", "kind"])[["acc", "fid", "nodes", "queries", "samples", "secs", "changed", "sem_splits"]].mean(); print(m.round(3).to_string())
rng = np.random.default_rng(20261120)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))
pairs = [("W", "O"), ("E", "W"), ("A", "W"), ("D", "O"), ("T", "O"), ("L", "O"), ("T", "Tq"), ("L", "Lq"), ("O", "B"), ("T", "B"), ("L", "B"), ("W", "B")]
S = pd.DataFrame([dict(contraste=f"{met}: {x} − {y}", x=x, y=y, metrica=met, dataset=ds, **ci((T[x] - T[y]).loc[ds]))
                  for x, y in pairs for met, T in (("accuracy", A), ("fidelidade", F)) for ds in DS])
S.to_csv(OUTC, index=False); print(S.drop(columns=["x", "y", "metrica"]).to_string())
# ---- regra de escolha (FASE5_PREREGISTO §5)
sel = []
for c in ("W", "T", "L"):
    s = S[(S.x == c) & (S.y == "O")]
    eligible = bool((s.hi >= 0).all())          # nenhum intervalo inteiramente abaixo de zero
    score = float(((A[c] - A.O) + (F[c] - F.O)).mean() / 2)
    sel.append(dict(candidata=c, elegivel=eligible, media_delta=round(score, 5), datasets_abaixo=s[s.hi < 0][["dataset", "metrica"]].values.tolist()))
sel = pd.DataFrame(sel); print(sel.to_string())
el = sel[sel.elegivel & (sel.media_delta > 0)]
order = {"O": 0, "T": 1, "L": 2, "W": 3}
if el.empty:
    chosen = "O"
else:
    best = el.media_delta.max(); tied = el[el.media_delta >= best - 0.001]
    chosen = sorted(tied.candidata, key=order.get)[0]
print("ESCOLHIDA:", chosen)
# não-inferioridade face a B
ni = S[(S.x == chosen) & (S.y == "B")] if chosen != "O" else S[(S.x == "O") & (S.y == "B")]
ni = ni.assign(nao_inferior=(ni.media >= 0) & (ni.lo >= -0.02)); print(ni.drop(columns=["x", "y"]).to_string())
json.dump({"escolhida": chosen, "candidatas": sel.to_dict("records"), "integridade": integ,
           "nao_inferioridade_vs_B": ni.drop(columns=["x", "y"]).to_dict("records")}, open(OUTSEL, "w"), indent=1, ensure_ascii=False)
fmt = lambda r: f"{r.media:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]".replace(".", ",")
hdr = "| Contraste | iris | wine | breast_cancer |\n|---|---|---|---|"
tab = lambda ks: "\n".join([hdr] + ["| " + k + " | " + " | ".join(fmt(S[(S.contraste == k) & (S.dataset == ds)].iloc[0]) for ds in DS) + " |" for k in ks])
key = lambda x, y, met: f"{met}: {x} − {y}"
T1 = tab([key(x, y, met) for x, y in pairs[:4] for met in ("accuracy", "fidelidade")])
T2 = tab([key(x, y, met) for x, y in pairs[4:8] for met in ("accuracy", "fidelidade")])
T3 = tab([key(x, y, met) for x, y in pairs[8:] for met in ("accuracy", "fidelidade")])
SEL = "\n".join(["| Candidata | Elegível | Média de (Δacc + Δfid)/2 face a O | Intervalos abaixo de zero |", "|---|---|---|---|"] +
                [f"| {r.candidata} | {'sim' if r.elegivel else 'não'} | {r.media_delta:+.4f} | {', '.join(f'{a} ({b})' for a, b in r.datasets_abaixo) or '—'} |".replace(".", ",") for r in sel.itertuples()])
f_ = lambda v, n=3: "N/A" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")
L = ["| Dataset | Braço | Accuracy | Fidelidade | Nós | Consultas da árvore final | Decisões alteradas pela semântica | Tempo (s) |", "|---|---|---|---|---|---|---|---|"]
for ds in DS:
    for k in ("C4.5", "B", "O", "W", "E", "A", "D", "T", "L", "Tq", "Lq"):
        r = m.loc[(ds, k)]; L.append(f"| {ds} | {k} | {f_(r.acc)} | {f_(r.fid)} | {f_(r.nodes, 1)} | {f_(r.queries, 0)} | {f_(r.changed, 1)} | {f_(r.secs, 1)} |")
open(OUTT, "w").write("\n@@\n".join([T1, T2, T3, SEL, "\n".join(L), chosen, json.dumps(integ)]))
