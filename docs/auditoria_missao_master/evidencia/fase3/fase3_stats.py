import json, glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
SC, CT, F2, OUTU, OUTC, OUTT = sys.argv[1:7]
DS = ["iris", "wine", "breast_cancer"]
def load(root):
    rows = []
    for f in sorted(glob.glob(root + "/raw/*/seed*/*.json")):
        b = os.path.basename(f)[:-5]
        if b in ("split", "UNIT_SHA256"): continue
        r = json.load(open(f))["row"]
        rows.append(dict(dataset=r["dataset"], seed=int(r["seed"]), arm=b, acc=r["accuracy_real_labels"], fid=r.get("fidelity_to_oracle"), nodes=r.get("node_count"),
                         secs=r.get("arm_time_s"), samples=r.get("total_oracle_samples") or r.get("membership_queries"), oracle_id=r.get("oracle_id"),
                         derived=r.get("semantic_features_generated"), used=r.get("semantic_features_used")))
    return pd.DataFrame(rows)
d = pd.concat([load(SC), (lambda m: m[m.arm.str.startswith("m29_")])(load(CT))], ignore_index=True)
def cond(a):
    return ("C4.5" if a == "c45" else "MLP" if a == "mlp_original" else "B Original + conjunto" if a.startswith("sc_B_") else
            "D2 Reloaded+OWL v2 + conjunto" if a.startswith("sc_D_") else "F Reloaded sem OWL + conjunto" if a.startswith("sc_F_") else
            "P2 permutada" if a.startswith("m29_P_") else "Dg2 grupos aleatórios" if a.startswith("m29_D_") else
            "Ec2 eng. convencional" if a.startswith("m29_E_") else "G2 Original, consultas igualadas" if a.startswith("m29_G_") else None)
d["cond"] = d.arm.map(cond); d = d[d.cond.notna()]
print("unidades:", d.groupby("dataset").seed.nunique().to_dict(), "| réplicas:", d.groupby(["cond", "dataset", "seed"]).size().groupby("cond").agg(["min", "max"]).to_dict("index"))
u = d.groupby(["dataset", "seed", "cond"])[["acc", "fid", "nodes", "secs", "samples", "derived", "used"]].mean().reset_index()
f2 = pd.read_csv(F2); f2 = f2[f2.dataset.isin(DS)]
d1 = f2[f2.cond == "D Reloaded+OWL + conjunto"].assign(cond="D1 Reloaded+OWL v1 + conjunto (fase 2)")
b2 = f2[f2.cond == "B Original + conjunto"].assign(cond="B fase 2")
u = pd.concat([u, d1[["dataset", "seed", "cond", "acc", "fid", "nodes", "secs", "samples"]], b2[["dataset", "seed", "cond", "acc", "fid", "nodes", "secs", "samples"]]], ignore_index=True)
u.to_csv(OUTU, index=False)
A = u.pivot_table(index=["dataset", "seed"], columns="cond", values="acc"); F = u.pivot_table(index=["dataset", "seed"], columns="cond", values="fid")
integ = float(max((A["B Original + conjunto"] - A["B fase 2"]).abs().max(), (F["B Original + conjunto"] - F["B fase 2"]).abs().max()))
print("INTEGRIDADE: diferença máxima de B entre fase 3 e fase 2 =", integ)
m = u.groupby(["dataset", "cond"])[["acc", "fid", "nodes", "samples", "secs", "derived", "used"]].mean()
print(m.round(3).to_string())
rng = np.random.default_rng(20261118)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))
B, D2, D1 = "B Original + conjunto", "D2 Reloaded+OWL v2 + conjunto", "D1 Reloaded+OWL v1 + conjunto (fase 2)"
con = {"H1 accuracy: D2 − B": (A, D2, B), "H2 accuracy: D2 − C4.5": (A, D2, "C4.5"), "H3 accuracy: B − C4.5": (A, B, "C4.5"), "H4 fidelidade: D2 − B": (F, D2, B),
       "H5 fidelidade: D2 − C4.5": (F, D2, "C4.5"), "K1 accuracy: D2 − D1": (A, D2, D1), "K1 fidelidade: D2 − D1": (F, D2, D1)}
for k, n in (("F", "F Reloaded sem OWL + conjunto"), ("P2", "P2 permutada"), ("Dg2", "Dg2 grupos aleatórios"), ("Ec2", "Ec2 eng. convencional"), ("G2", "G2 Original, consultas igualadas")):
    con[f"accuracy: D2 − {k}"] = (A, D2, n); con[f"fidelidade: D2 − {k}"] = (F, D2, n)
S = pd.DataFrame([dict(contraste=k, dataset=ds, **ci((T[x] - T[y]).loc[ds])) for k, (T, x, y) in con.items() for ds in DS])
S.to_csv(OUTC, index=False); print(S.to_string())
H = pd.DataFrame({"H1": A[D2] > A[B], "H2": A[D2] > A["C4.5"], "H3": A[B] > A["C4.5"], "H4": F[D2] > F[B], "H5": F[D2] > F["C4.5"]}); H["conjunta"] = H.all(axis=1)
print(H.groupby(level=0).sum().to_string()); print("total", len(H), H.sum().to_dict()); print(H[H.conjunta])
sd = d.groupby(["dataset", "seed", "cond"])[["acc", "fid"]].std().groupby("dataset").mean().round(4); print("dp réplicas\n", sd.to_string())
# tabelas markdown
fmt = lambda r: f"{r.media:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]".replace(".", ",")
hdr = "| Contraste | iris | wine | breast_cancer |\n|---|---|---|---|"
def tab(keys): return "\n".join([hdr] + ["| " + k + " | " + " | ".join(fmt(S[(S.contraste == k) & (S.dataset == ds)].iloc[0]) for ds in DS) + " |" for k in keys])
keys = list(con)
order = ["MLP", "C4.5", B, "G2 Original, consultas igualadas", D1, D2, "F Reloaded sem OWL + conjunto", "P2 permutada", "Dg2 grupos aleatórios", "Ec2 eng. convencional"]
f = lambda v, n=3: "N/A" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")
L = ["| Dataset | Condição | Accuracy | Fidelidade | Nós | Amostras do oráculo | Tempo (s) |", "|---|---|---|---|---|---|---|"]
for ds in DS:
    for c in order:
        r = m.loc[(ds, c)]; L.append(f"| {ds} | {c} | {f(r.acc)} | {f(r.fid)} | {f(r.nodes, 1)} | {f(r.samples, 0)} | {f(r.secs, 1)} |")
hh = H.groupby(level=0).sum()
HT = "\n".join(["| Dataset | H1 | H2 | H3 | H4 | H5 | Conjunta |", "|---|---|---|---|---|---|---|"] + [f"| {ds} | " + " | ".join(f"{int(hh.loc[ds, c])}/5" for c in hh.columns) + " |" for ds in DS]
               + ["| **Total** | " + " | ".join(f"{int(H[c].sum())}/{len(H)}" for c in H.columns) + " |"])
open(OUTT, "w").write("\n@@\n".join([tab(keys[:5]), tab(keys[5:7]), tab(keys[7:]), "\n".join(L), HT]))
