import json, glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
F7, F6, ELIG, OUTU, OUTC, OUTT = sys.argv[1:7]
DS = ["german_credit", "phys_bmi", "phys_kinetic", "phys_power"]
def load(root, tag):
    rows = []
    for f in sorted(glob.glob(root + "/raw/*/seed*/*.json")):
        b = os.path.basename(f)[:-5]
        if b in ("split", "UNIT_SHA256"): continue
        j = json.load(open(f)); r = j["row"]
        kind = b.split("_")[1] if b.startswith("sc_") else {"c45": "C4.5", "mlp_original": "MLP"}.get(b)
        if kind is None: continue
        rows.append(dict(campanha=tag, dataset=r["dataset"], seed=int(r["seed"]), kind=kind, acc=r["accuracy_real_labels"], f1=r.get("macro_f1_real_labels"),
                         bacc=r.get("balanced_accuracy_real_labels"), fid=r.get("fidelity_to_oracle"), nodes=r.get("node_count"), depth=r.get("depth"),
                         queries=r.get("membership_queries"), secs=r.get("arm_time_s"), test_idx=tuple(j["predictions"]["test_row_index"])))
    return pd.DataFrame(rows)
d7, d6 = load(F7, "fase7"), load(F6, "fase6")
# mesmas partições?
p7 = d7[d7.kind == "MLP"].set_index(["dataset", "seed"]).test_idx; p6 = d6[d6.kind == "MLP"].set_index(["dataset", "seed"]).test_idx
same = bool((p7.sort_index() == p6.sort_index()).all()); print("mesmas partições de teste que a fase 6:", same)
print("unidades:", d7.groupby("dataset").seed.nunique().to_dict(), "| réplicas:", d7.groupby(["kind", "dataset", "seed"]).size().groupby("kind").agg(["min", "max"]).to_dict("index"))
cols = ["acc", "f1", "bacc", "fid", "nodes", "depth", "queries", "secs"]
u = d7.groupby(["dataset", "seed", "kind"])[cols].mean().reset_index()
mo = d6[d6.kind == "MLP"][["dataset", "seed", "acc", "f1", "bacc"]].assign(kind="MLP_orig")
u = pd.concat([u, mo], ignore_index=True); u.to_csv(OUTU, index=False)
A = u.pivot_table(index=["dataset", "seed"], columns="kind", values="acc"); F = u.pivot_table(index=["dataset", "seed"], columns="kind", values="fid")
F1 = u.pivot_table(index=["dataset", "seed"], columns="kind", values="f1"); BA = u.pivot_table(index=["dataset", "seed"], columns="kind", values="bacc")
m = u.groupby(["dataset", "kind"])[cols].mean(); print(m.round(3).to_string())
el = json.load(open(ELIG))["per_dataset"]; print("elegibilidade:", el)
rng = np.random.default_rng(20261122)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))
rows = []
for met, T in (("accuracy", A), ("macro-F1", F1), ("balanced accuracy", BA)):
    for x, y in (("MLP", "MLP_orig"), ("MLP", "C4.5"), ("MLP_orig", "C4.5")):
        for ds in DS:
            rows.append(dict(orcamento="—", id="M", contraste=f"{met}: {x.replace('MLP_orig','MLP original').replace('MLP','MLP otimizado') if x=='MLP' else 'MLP original'} − {y.replace('MLP_orig','MLP original')}", dataset=ds, **ci((T[x] - T[y]).loc[ds])))
for suf, lab in (("", "estrutura do tuning"), ("7", "máx. 7 nós")):
    R_, B_, Q_ = "R" + suf, "B" + suf, "Rq" + suf
    for cid, met, T, x, y in (("H1", "accuracy", A, R_, B_), ("H2", "accuracy", A, R_, "C4.5"), ("H3", "accuracy", A, B_, "C4.5"),
                              ("H4", "fidelidade", F, R_, B_), ("H5", "fidelidade", F, R_, "C4.5"), ("AT", "accuracy", A, R_, Q_), ("AT", "fidelidade", F, R_, Q_)):
        for ds in DS:
            rows.append(dict(orcamento=lab, id=cid, contraste=f"{cid} {met}: {x} − {y}", dataset=ds, **ci((T[x] - T[y]).loc[ds])))
S = pd.DataFrame(rows); S.to_csv(OUTC, index=False); print(S.to_string())
verdict = []
for lab in ("estrutura do tuning", "máx. 7 nós"):
    for ds in DS:
        s = S[(S.orcamento == lab) & (S.dataset == ds)]; h = s[s.id.isin(["H1", "H2", "H3", "H4", "H5"])]; at = s[s.id == "AT"]
        verdict.append(dict(orcamento=lab, dataset=ds, elegivel=el[ds]["eligible"], sustentada=bool((h.lo > 0).all()), observada=bool((h.media > 0).all()),
                            atribuivel=bool((s[s.id.isin(["H1", "H4"])].lo > 0).all() and (at.lo > 0).all())))
V = pd.DataFrame(verdict); print(V.to_string())
units = []
for suf in ("", "7"):
    H = pd.DataFrame({"H1": A["R" + suf] > A["B" + suf], "H2": A["R" + suf] > A["C4.5"], "H3": A["B" + suf] > A["C4.5"], "H4": F["R" + suf] > F["B" + suf], "H5": F["R" + suf] > F["C4.5"]})
    H["conjunta"] = H.all(axis=1); units.append(H.groupby(level=0).sum().assign(orcamento="máx. 7 nós" if suf else "estrutura do tuning"))
U = pd.concat(units); print(U.to_string())
# configurações escolhidas
ch = []
for f in sorted(glob.glob(F7 + "/raw/*/seed*/split.json")):
    pass
fmt = lambda r: f"{r.media:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]".replace(".", ",")
def tab(sel):
    s = S[sel]; out = ["| Contraste | " + " | ".join(DS) + " |", "|---" * (len(DS) + 1) + "|"]
    for c in s.contraste.unique():
        out.append("| " + c + " | " + " | ".join(fmt(s[(s.contraste == c) & (s.dataset == ds)].iloc[0]) for ds in DS) + " |")
    return "\n".join(out)
f_ = lambda v, n=3: "N/A" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")
L = ["| Dataset | Braço | Accuracy | Macro-F1 | Balanced acc. | Fidelidade | Nós | Profundidade | Tempo (s) |", "|---|---|---|---|---|---|---|---|---|"]
for ds in DS:
    for k in ("MLP_orig", "MLP", "C4.5", "B", "R", "Rq", "B7", "R7", "Rq7"):
        r = m.loc[(ds, k)]; nm = {"MLP_orig": "MLP original (fase 6)", "MLP": "MLP otimizado"}.get(k, k)
        L.append(f"| {ds} | {nm} | {f_(r.acc)} | {f_(r.f1)} | {f_(r.bacc)} | {f_(r.fid)} | {f_(r.nodes, 1)} | {f_(r.depth, 1)} | {f_(r.secs, 1)} |")
VT = "\n".join(["| Orçamento | Dataset | Elegível | Sustentada | Observada | Atribuível |", "|---|---|---|---|---|---|"] +
               [f"| {r.orcamento} | {r.dataset} | {'sim' if r.elegivel else 'não'} | {'sim' if r.sustentada else 'não'} | {'sim' if r.observada else 'não'} | {'sim' if r.atribuivel else 'não'} |" for r in V.itertuples()])
UT = "\n".join(["| Orçamento | Dataset | H1 | H2 | H3 | H4 | H5 | Conjunta |", "|---|---|---|---|---|---|---|---|"] +
               [f"| {r.orcamento} | {ds} | " + " | ".join(f"{int(getattr(r, c))}/5" for c in ("H1", "H2", "H3", "H4", "H5", "conjunta")) + " |" for ds, r in U.iterrows()])
ET = "\n".join(["| Dataset | MLP otimizado − C4.5 (CV aninhada no treino) | Elegível |", "|---|---|---|"] + [f"| {ds} | {el[ds]['mean_gap']:+.3f} | {'sim' if el[ds]['eligible'] else 'não'} |".replace(".", ",") for ds in DS])
open(OUTT, "w").write("\n@@\n".join([tab(S.id == "M"), tab(S.orcamento == "estrutura do tuning"), tab(S.orcamento == "máx. 7 nós"), VT, UT, "\n".join(L), ET, str(same)]))
