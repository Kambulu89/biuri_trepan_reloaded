import json, glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
SC, M29 = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
def load(root):
    rows = []
    for f in sorted(glob.glob(root + "/raw/*/seed*/*.json")):
        b = os.path.basename(f)[:-5]
        if b in ("split", "UNIT_SHA256"): continue
        r = json.load(open(f))["row"]
        rows.append(dict(dataset=r["dataset"], seed=int(r["seed"]), arm=b, acc=r["accuracy_real_labels"], fid=r.get("fidelity_to_oracle"), f1=r.get("macro_f1_real_labels"),
                         nodes=r.get("node_count"), rules=r.get("rule_count"), depth=r.get("depth"), secs=r.get("arm_time_s"), oracle_id=r.get("oracle_id"),
                         samples=r.get("total_oracle_samples") or r.get("membership_queries")))
    return pd.DataFrame(rows)
d = load(SC)
if M29 and os.path.isdir(M29 + "/raw"):
    m = load(M29); d = pd.concat([d, m[m.arm.str.startswith("m29_")]], ignore_index=True)
def cond(a):
    if a == "c45": return "C4.5"
    if a == "mlp_original": return "MLP"
    if a == "trepan_original" or a.startswith("sc_A_"): return "A Original (marginal)"
    if a.startswith("sc_B_"): return "B Original + conjunto"
    if a == "reloaded_owl_full" or a.startswith("sc_C_"): return "C Reloaded+OWL (marginal)"
    if a.startswith("sc_D_"): return "D Reloaded+OWL + conjunto"
    if a == "reloaded_core" or a.startswith("sc_E_"): return "E Reloaded sem OWL (marginal)"
    if a.startswith("sc_F_"): return "F Reloaded sem OWL + conjunto"
    if a.startswith("m29_P_"): return "P permutada + conjunto"
    if a.startswith("m29_D_"): return "Dg grupos aleatórios + conjunto"
    if a.startswith("m29_E_"): return "Ec eng. convencional + conjunto"
    if a.startswith("m29_G_"): return "G Original + conjunto, consultas igualadas"
    return None
d["cond"] = d.arm.map(cond); d = d[d.cond.notna()]
print("réplicas por condição/unidade:", d.groupby(["cond", "dataset", "seed"]).size().groupby("cond").agg(["min", "max"]).to_dict("index"))
u = d.groupby(["dataset", "seed", "cond"])[["acc", "fid", "f1", "nodes", "rules", "depth", "secs", "samples"]].mean().reset_index()
u.to_csv(sys.argv[3], index=False)
print(u.groupby(["dataset", "cond"])[["acc", "fid", "nodes", "samples", "secs"]].mean().round(3).to_string())
rep_sd = d.groupby(["dataset", "seed", "cond"])[["acc", "fid"]].std().groupby("dataset").mean().round(4); print("dp entre réplicas:\n", rep_sd.to_string())
A = u.pivot_table(index=["dataset", "seed"], columns="cond", values="acc"); F = u.pivot_table(index=["dataset", "seed"], columns="cond", values="fid")
rng = np.random.default_rng(20261117)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))
B, D = "B Original + conjunto", "D Reloaded+OWL + conjunto"
con = {"H1 acc D−B": (A, D, B), "H2 acc D−C4.5": (A, D, "C4.5"), "H3 acc B−C4.5": (A, B, "C4.5"), "H4 fid D−B": (F, D, B), "H5 fid D−C4.5": (F, D, "C4.5"),
       "acc D−F": (A, D, "F Reloaded sem OWL + conjunto"), "fid D−F": (F, D, "F Reloaded sem OWL + conjunto"),
       "amostrador acc B−A": (A, B, "A Original (marginal)"), "amostrador fid B−A": (F, B, "A Original (marginal)"),
       "amostrador acc D−C": (A, D, "C Reloaded+OWL (marginal)"), "amostrador fid D−C": (F, D, "C Reloaded+OWL (marginal)"),
       "fid B−C4.5": (F, B, "C4.5"), "acc MLP−C4.5": (A, "MLP", "C4.5")}
for k, name in (("P", "P permutada + conjunto"), ("Dg", "Dg grupos aleatórios + conjunto"), ("Ec", "Ec eng. convencional + conjunto"), ("G", "G Original + conjunto, consultas igualadas")):
    if name in A.columns:
        con[f"acc D−{k}"] = (A, D, name); con[f"fid D−{k}"] = (F, D, name)
S = []
for k, (T, x, y) in con.items():
    for ds in ["iris", "wine", "breast_cancer", "digits"]:
        S.append(dict(contraste=k, dataset=ds, **ci((T[x] - T[y]).loc[ds])))
S = pd.DataFrame(S); S.to_csv(sys.argv[4], index=False); print(S.to_string())
H = pd.DataFrame({"H1": A[D] > A[B], "H2": A[D] > A["C4.5"], "H3": A[B] > A["C4.5"], "H4": F[D] > F[B], "H5": F[D] > F["C4.5"]}); H["conjunta"] = H.all(axis=1)
print(H.groupby(level=0).sum().to_string()); print("total", len(H), H.sum().to_dict())
print(H[H.conjunta])
