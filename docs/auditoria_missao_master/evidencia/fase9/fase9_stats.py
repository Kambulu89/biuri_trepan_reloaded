import json, glob, os, sys, numpy as np, pandas as pd
pd.set_option("display.width", 250)
F9, ELIG7, ELIG8, PRIOR7, PRIOR8, OUTU, OUTC, OUTT, OUTI = sys.argv[1:10]
DS_ALL = ["digits", "german_credit", "phys_bmi", "phys_power", "phys_kinetic", "adult"]
PRIOR = {"digits": PRIOR8, "adult": PRIOR8, "german_credit": PRIOR7, "phys_bmi": PRIOR7, "phys_power": PRIOR7, "phys_kinetic": PRIOR7}


def load(root):
    rows, preds = [], {}
    for f in sorted(glob.glob(root + "/raw/*/seed*/*.json")):
        b = os.path.basename(f)[:-5]
        if b in ("split", "UNIT_SHA256"): continue
        j = json.load(open(f)); r = j["row"]
        preds[(r["dataset"], int(r["seed"]), b)] = j["predictions"]
        kind = b.split("_")[1] if b.startswith("sc_") else {"c45": "C4.5", "mlp_original": "MLP"}.get(b)
        if kind is None: continue
        rows.append(dict(dataset=r["dataset"], seed=int(r["seed"]), arm=b, kind=kind, acc=r["accuracy_real_labels"], f1=r.get("macro_f1_real_labels"),
                         bacc=r.get("balanced_accuracy_real_labels"), fid=r.get("fidelity_to_oracle"), nodes=r.get("node_count"), depth=r.get("depth"),
                         queries=r.get("oracle_queries"), secs=r.get("arm_time_s")))
    return pd.DataFrame(rows), preds


d, P = load(F9)
DS = [x for x in DS_ALL if x in set(d.dataset)]
print("unidades:", d.groupby("dataset").seed.nunique().to_dict(), "| réplicas:", d.groupby(["kind", "dataset", "seed"]).size().groupby("kind").agg(["min", "max"]).to_dict("index"))

# ---- integridade 1: MLP, C4.5, B, R iguais às fases 7/8 na mesma unidade
integ = {"igual_fase_anterior": {}, "E_igual_V_sem_tentativas": {}}
cache = {}
for (ds, s, arm), pr in P.items():
    if not (arm in ("mlp_original", "c45") or (arm.startswith("sc_") and arm.split("_")[1] in ("B", "R", "B7", "R7"))):
        continue
    f = f"{PRIOR[ds]}/raw/{ds}/seed{s}/{arm}.json"
    ok = os.path.exists(f) and json.load(open(f))["predictions"] == pr
    integ["igual_fase_anterior"].setdefault(ds, {"iguais": 0, "diferentes": 0})["iguais" if ok else "diferentes"] += 1
# ---- integridade 2 e diagnóstico do EFSR
aud = []
for f in sorted(glob.glob(F9 + "/audits/*__fase9_efsr.jsonl")):
    ds, s = os.path.basename(f).split("__")[0].rsplit("_seed", 1)
    for line in open(f):
        a = json.loads(line)
        if not str(a.get("arm") or "").startswith("sc_"): continue
        aud.append(dict(dataset=ds, seed=int(s), arm=a["arm"], kind=a["arm"].split("_")[1], attempts=a["attempts"], accepted_v=a["accepted"],
                        holdout_queries=a["holdout_queries"], efsr_attempted=a["efsr_attempted"], efsr_accepted=a["efsr_accepted"]))
AU = pd.DataFrame(aud)
for r in AU[AU.kind.isin(["V", "V7"]) & (AU.attempts == 0)].itertuples():
    e_arm = r.arm.replace("_V", "_E", 1)
    ok = P.get((r.dataset, r.seed, r.arm)) == P.get((r.dataset, r.seed, e_arm))
    integ["E_igual_V_sem_tentativas"].setdefault(r.dataset, {"iguais": 0, "diferentes": 0})["iguais" if ok else "diferentes"] += 1
json.dump(integ, open(OUTI, "w"), indent=1); print("integridade:", integ)
diag = AU.groupby(["dataset", "kind"])[["efsr_attempted", "efsr_accepted", "holdout_queries"]].mean()
print(diag.round(2).to_string())

cols = ["acc", "f1", "bacc", "fid", "nodes", "depth", "queries", "secs"]
u = d.groupby(["dataset", "seed", "kind"])[cols].mean().reset_index(); u.to_csv(OUTU, index=False)
A = u.pivot_table(index=["dataset", "seed"], columns="kind", values="acc"); F = u.pivot_table(index=["dataset", "seed"], columns="kind", values="fid")
m = u.groupby(["dataset", "kind"])[cols].mean()
el = {**json.load(open(ELIG7))["per_dataset"], **json.load(open(ELIG8))["per_dataset"]}; print("elegibilidade:", {k: el[k] for k in DS})
rng = np.random.default_rng(20261125)
def ci(x):
    x = np.asarray(x, float); b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))
rows = []
for suf, lab in (("", "estrutura do tuning"), ("7", "máx. 7 nós")):
    V_, B_, Q_, E_, R_ = "V" + suf, "B" + suf, "Vq" + suf, "E" + suf, "R" + suf
    for cid, met, T, x, y in (("H1", "accuracy", A, V_, B_), ("H2", "accuracy", A, V_, "C4.5"), ("H3", "accuracy", A, B_, "C4.5"),
                              ("H4", "fidelidade", F, V_, B_), ("H5", "fidelidade", F, V_, "C4.5"), ("AT", "accuracy", A, V_, Q_), ("AT", "fidelidade", F, V_, Q_),
                              ("G1", "accuracy", A, V_, E_), ("G1", "fidelidade", F, V_, E_), ("G2", "accuracy", A, V_, R_), ("G2", "fidelidade", F, V_, R_),
                              ("E-R", "accuracy", A, E_, R_), ("E-R", "fidelidade", F, E_, R_)):
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
# regra de adoção (estrutura do tuning, só elegíveis)
g2 = S[(S.orcamento == "estrutura do tuning") & (S.id == "G2") & S.dataset.map(lambda x: el[x]["eligible"])]
adopt = bool((g2.lo > 0).any() and not (g2.hi < 0).any())
adopt_s = f"regra de adoção: G2 com limite inferior > 0 em {sorted(set(g2[g2.lo > 0].dataset))}; G2 com limite superior < 0 em {sorted(set(g2[g2.hi < 0].dataset))}; útil = {adopt}"
print(adopt_s)
units = []
for suf in ("", "7"):
    H = pd.DataFrame({"H1": A["V" + suf] > A["B" + suf], "H2": A["V" + suf] > A["C4.5"], "H3": A["B" + suf] > A["C4.5"], "H4": F["V" + suf] > F["B" + suf], "H5": F["V" + suf] > F["C4.5"]})
    H["conjunta"] = H.all(axis=1); units.append(H.groupby(level=0).sum().assign(orcamento="máx. 7 nós" if suf else "estrutura do tuning"))
U = pd.concat(units); print(U.to_string())
fmt = lambda r: f"{r.media:+.3f} [{r.lo:+.3f}; {r.hi:+.3f}]".replace(".", ",")
def tab(sel):
    s = S[sel]; out = ["| Contraste | " + " | ".join(DS) + " |", "|---" * (len(DS) + 1) + "|"]
    for c in s.contraste.unique():
        out.append("| " + c + " | " + " | ".join(fmt(s[(s.contraste == c) & (s.dataset == ds)].iloc[0]) for ds in DS) + " |")
    return "\n".join(out)
f_ = lambda v, n=3: "N/A" if pd.isna(v) else f"{v:.{n}f}".replace(".", ",")
L = ["| Dataset | Braço | Accuracy | Macro-F1 | Balanced acc. | Fidelidade | Nós | Consultas ao oráculo | Tempo (s) |", "|---|---|---|---|---|---|---|---|---|"]
for ds in DS:
    for k in ("MLP", "C4.5", "B", "R", "E", "V", "Vq", "B7", "R7", "E7", "V7", "Vq7"):
        r = m.loc[(ds, k)]
        L.append(f"| {ds} | {k} | {f_(r.acc)} | {f_(r.f1)} | {f_(r.bacc)} | {f_(r.fid)} | {f_(r.nodes, 1)} | {f_(r.queries, 0)} | {f_(r.secs, 1)} |")
DG = ["| Dataset | Braço | Tentativas do EFSR por árvore | Aprovações por árvore | Consultas do lote novo por árvore |", "|---|---|---|---|---|"]
for (ds, k), r in diag.iterrows():
    if k.rstrip("7") in ("E", "V", "Vq"):
        DG.append(f"| {ds} | {k} | {f_(r.efsr_attempted, 1)} | {f_(r.efsr_accepted, 1)} | {f_(r.holdout_queries, 0)} |")
VT = "\n".join(["| Orçamento | Dataset | Elegível | Sustentada | Observada | Atribuível |", "|---|---|---|---|---|---|"] +
               [f"| {r.orcamento} | {r.dataset} | {'sim' if r.elegivel else 'não'} | {'sim' if r.sustentada else 'não'} | {'sim' if r.observada else 'não'} | {'sim' if r.atribuivel else 'não'} |" for r in V.itertuples()])
UT = "\n".join(["| Orçamento | Dataset | H1 | H2 | H3 | H4 | H5 | Conjunta |", "|---|---|---|---|---|---|---|---|"] +
               [f"| {r.orcamento} | {ds} | " + " | ".join(f"{int(getattr(r, c))}/5" for c in ("H1", "H2", "H3", "H4", "H5", "conjunta")) + " |" for ds, r in U.iterrows()])
open(OUTT, "w").write("\n@@\n".join([tab(S.orcamento == "estrutura do tuning"), tab(S.orcamento == "máx. 7 nós"), VT, UT, "\n".join(L), "\n".join(DG), adopt_s, json.dumps(integ, ensure_ascii=False)]))
