import json, glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
F4, F3U, OUTU, OUTC, OUTT = sys.argv[1:6]
DS = ["iris", "wine", "breast_cancer"]
rows = []
for f in sorted(glob.glob(F4 + "/raw/*/seed*/sc_*.json")):
    b = os.path.basename(f)[:-5]; r = json.load(open(f))["row"]
    rows.append(dict(dataset=r["dataset"], seed=int(r["seed"]), arm=b, kind=b.split("_")[1], acc=r["accuracy_real_labels"], fid=r["fidelity_to_oracle"],
                     nodes=r.get("node_count"), queries=r.get("membership_queries"), samples=r.get("total_oracle_samples") or r.get("membership_queries"),
                     secs=r.get("arm_time_s"), used=r.get("semantic_features_used")))
d = pd.DataFrame(rows)
print("unidades:", d.groupby("dataset").seed.nunique().to_dict(), "| réplicas:", d.groupby(["kind", "dataset", "seed"]).size().groupby("kind").agg(["min", "max"]).to_dict("index"))
u = d.groupby(["dataset", "seed", "kind"])[["acc", "fid", "nodes", "queries", "samples", "secs", "used"]].mean().reset_index()
f3 = pd.read_csv(F3U); f3 = f3[f3.dataset.isin(DS)]
take = {"C4.5": "C4.5", "D2 Reloaded+OWL v2 + conjunto": "D2", "B Original + conjunto": "B3", "MLP": "MLP"}
f3 = f3[f3.cond.isin(take)].assign(kind=lambda x: x.cond.map(take))
u = pd.concat([u, f3[["dataset", "seed", "kind", "acc", "fid", "nodes", "samples", "secs"]]], ignore_index=True)
u.to_csv(OUTU, index=False)
A = u.pivot_table(index=["dataset", "seed"], columns="kind", values="acc"); F = u.pivot_table(index=["dataset", "seed"], columns="kind", values="fid")
integ = float(max((A.B - A.B3).abs().max(), (F.B - F.B3).abs().max())); print("INTEGRIDADE B fase4 vs fase3:", integ)
m = u.groupby(["dataset", "kind"])[["acc", "fid", "nodes", "queries", "samples", "secs", "used"]].mean(); print(m.round(3).to_string())
rng = np.random.default_rng(20261119)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))
con = {}
for lab, x, y in (("J1", "O", "B"), ("J2", "O", "Q"), ("J2", "O", "K"), ("J3", "D2", "O"), ("H", "O", "C4.5"), ("H", "B", "C4.5")):
    for met, T in (("accuracy", A), ("fidelidade", F)):
        con[f"{lab} {met}: {x} − {y}"] = (T, x, y)
S = pd.DataFrame([dict(contraste=k, dataset=ds, **ci((T[x] - T[y]).loc[ds])) for k, (T, x, y) in con.items() for ds in DS])
S.to_csv(OUTC, index=False); print(S.to_string())
H = pd.DataFrame({"O>B acc": A.O > A.B, "O>C4.5 acc": A.O > A["C4.5"], "B>C4.5 acc": A.B > A["C4.5"], "O>B fid": F.O > F.B, "O>C4.5 fid": F.O > F["C4.5"]})
H["conjunta"] = H.all(axis=1); print(H.groupby(level=0).sum().to_string()); print("total", len(H), H.sum().to_dict())
sd = d.groupby(["dataset", "seed", "kind"])[["acc", "fid"]].std().groupby("dataset").mean().round(4); print("dp réplicas\n", sd.to_string())
fmt = lambda r: f"{r.media:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]".replace(".", ",")
hdr = "| Contraste | iris | wine | breast_cancer |\n|---|---|---|---|"
tab = lambda keys: "\n".join([hdr] + ["| " + k + " | " + " | ".join(fmt(S[(S.contraste == k) & (S.dataset == ds)].iloc[0]) for ds in DS) + " |" for k in keys])
ks = list(con)
names = {"MLP": "MLP", "C4.5": "C4.5", "B": "B Original + conjunto", "O": "O Original-equivalente, espaço OWL v2", "Q": "Q O com OWL v2 permutada",
         "K": "K O com engenharia convencional", "D2": "D2 Reloaded + OWL v2 (fase 3)"}
f_ = lambda v, n=3: "N/A" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")
L = ["| Dataset | Braço | Accuracy | Fidelidade | Nós | Consultas da árvore final | Amostras do oráculo | Tempo (s) |", "|---|---|---|---|---|---|---|---|"]
for ds in DS:
    for k in ("MLP", "C4.5", "B", "O", "Q", "K", "D2"):
        r = m.loc[(ds, k)]; L.append(f"| {ds} | {names[k]} | {f_(r.acc)} | {f_(r.fid)} | {f_(r.nodes, 1)} | {f_(r.queries, 0)} | {f_(r.samples, 0)} | {f_(r.secs, 1)} |")
hh = H.groupby(level=0).sum()
HT = "\n".join(["| Dataset | " + " | ".join(H.columns) + " |", "|" + "---|" * (len(H.columns) + 1)] + [f"| {ds} | " + " | ".join(f"{int(hh.loc[ds, c])}/5" for c in H.columns) + " |" for ds in DS]
               + ["| **Total** | " + " | ".join(f"{int(H[c].sum())}/{len(H)}" for c in H.columns) + " |"])
open(OUTT, "w").write("\n@@\n".join([tab(ks[0:2]), tab(ks[2:6]), tab(ks[6:8]), tab(ks[8:12]), HT, "\n".join(L), f"{integ:.1e}"]))
