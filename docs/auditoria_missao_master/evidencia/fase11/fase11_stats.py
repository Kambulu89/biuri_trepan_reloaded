"""Análise da fase 11 (FASE11_PREREGISTO.md). Uso: python fase11_stats.py RAIZ_SELECIONADO RAIZ_OMISSAO ELIG7 ELIG8 OUT_DIR"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

SEL, OMI, EL7, EL8, OUT = sys.argv[1:6]
os.makedirs(OUT, exist_ok=True)
DS = ["iris", "wine", "breast_cancer", "digits", "adult", "german_credit", "phys_bmi", "phys_kinetic", "phys_power"]
PRIOR = {d: "/home/user/audit_runs/fase8" for d in ("iris", "wine", "breast_cancer", "digits", "adult")}
PRIOR.update({d: "/home/user/audit_runs/fase9" for d in ("german_credit", "phys_bmi", "phys_kinetic", "phys_power")})
SEEDS = (42, 7, 123, 2024, 11)
el = {**json.load(open(EL7))["per_dataset"], **json.load(open(EL8))["per_dataset"]}


def pred(root, ds, s, arm):
    p = f"{root}/raw/{ds}/seed{s}/{arm}.json"
    return json.load(open(p)) if os.path.exists(p) else None


# ------------------------------------------------------------------ integridade
integ = []
for root, prior, dsl in ((SEL, PRIOR, DS), (OMI, {d: "/home/user/audit_runs/fase6" for d in DS}, ["german_credit", "phys_bmi", "phys_kinetic", "phys_power"])):
    for ds in dsl:
        for s in SEEDS:
            for arm in ["mlp_original", "c45"] + [f"sc_{k}_r{r}" for k in ("B", "R", "B7", "R7") for r in range(3)]:
                a, b = pred(root, ds, s, arm), pred(prior[ds], ds, s, arm)
                integ.append(dict(execucao=os.path.basename(root), dataset=ds, seed=s, arm=arm,
                                  igual=a is not None and b is not None and a["predictions"] == b["predictions"]))
I = pd.DataFrame(integ)
I.to_csv(f"{OUT}/fase11_integridade.csv", index=False)
bad = set(map(tuple, I[~I.igual][["execucao", "dataset", "seed"]].drop_duplicates().values))

# ------------------------------------------------------------------ A. complexidade igual
rows = []
for ds in DS:
    for s in SEEDS:
        if ("fase11", ds, s) in bad or not os.path.exists(f"{SEL}/raw/{ds}/seed{s}/RUN_META.txt"):
            continue
        for f in glob.glob(f"{SEL}/raw/{ds}/seed{s}/*.json"):
            b = os.path.basename(f)[:-5]
            if b in ("split", "UNIT_SHA256"):
                continue
            kind = b.split("_")[1] if b.startswith("sc_") else {"c45": "C4.5", "c45_d2": "C4.5d2"}.get(b)
            if kind is None:
                continue
            r = json.load(open(f))["row"]
            rows.append(dict(dataset=ds, seed=s, kind=kind, acc=r["accuracy_real_labels"], fid=r.get("fidelity_to_oracle"), nodes=r.get("node_count"),
                             cap=r.get("tree_max_nodes")))
D = pd.DataFrame(rows)
u = D.groupby(["dataset", "seed", "kind"])[["acc", "fid", "nodes", "cap"]].agg({"acc": "mean", "fid": "mean", "nodes": "mean", "cap": "max"}).reset_index()
u.to_csv(f"{OUT}/fase11_por_unidade.csv", index=False)
A = u.pivot_table(index=["dataset", "seed"], columns="kind", values="acc")
F = u.pivot_table(index=["dataset", "seed"], columns="kind", values="fid")
N = u.pivot_table(index=["dataset", "seed"], columns="kind", values="nodes")
CAP = u.pivot_table(index=["dataset", "seed"], columns="kind", values="cap")
# verificação de construção
cons = []
for (ds, s), row in N.iterrows():
    n_cap = min(row["C4.5"], CAP.loc[(ds, s), "B"])
    cons.append(dict(dataset=ds, seed=s, c45_nos=row["C4.5"], teto_tuning=CAP.loc[(ds, s), "B"], N=n_cap,
                     Bc_ok=bool(D[(D.dataset == ds) & (D.seed == s) & (D.kind == "Bc")].nodes.max() <= n_cap),
                     Rc_ok=bool(D[(D.dataset == ds) & (D.seed == s) & (D.kind == "Rc")].nodes.max() <= n_cap),
                     d2_ok=bool(row["C4.5d2"] <= 7)))
K = pd.DataFrame(cons)
K.to_csv(f"{OUT}/fase11_construcao.csv", index=False)
rng = np.random.default_rng(20261121)


def ci(x):
    x = np.asarray(x, float)
    b = rng.choice(x, (20000, len(x))).mean(1)
    return dict(media=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4), pos=int((x > 1e-12).sum()), neg=int((x < -1e-12).sum()))


cs = []
for lab, items in (("tamanho do C4.5", (("H1c", "accuracy", A, "Rc", "Bc"), ("H2c", "accuracy", A, "Rc", "C4.5"), ("H3c", "accuracy", A, "Bc", "C4.5"),
                                         ("H4c", "fidelidade", F, "Rc", "Bc"), ("H5c", "fidelidade", F, "Rc", "C4.5"))),
                   ("7 nós", (("H1", "accuracy", A, "R7", "B7"), ("H2d", "accuracy", A, "R7", "C4.5d2"), ("H3d", "accuracy", A, "B7", "C4.5d2"),
                              ("H4", "fidelidade", F, "R7", "B7"), ("H5d", "fidelidade", F, "R7", "C4.5d2")))):
    for cid, met, T, x, y in items:
        for ds in DS:
            if ds in T.index.get_level_values(0):
                cs.append(dict(orcamento=lab, id=cid, contraste=f"{cid} {met}: {x} − {y}", dataset=ds, **ci((T[x] - T[y]).loc[ds])))
S = pd.DataFrame(cs)
S.to_csv(f"{OUT}/fase11_contrastes.csv", index=False)
V = []
for lab in ("tamanho do C4.5", "7 nós"):
    for ds in DS:
        s = S[(S.orcamento == lab) & (S.dataset == ds)]
        if s.empty:
            continue
        V.append(dict(orcamento=lab, dataset=ds, elegivel=bool(el[ds]["eligible"]), sustentada=bool((s.lo > 0).all()), observada=bool((s.media > 0).all()),
                      original_supera_c45=bool((s[s.id.isin(["H3c", "H3d"])].lo > 0).all()),
                      reloaded_supera_c45_acc_e_fid=bool((s[s.id.isin(["H2c", "H5c", "H2d", "H5d"])].lo > 0).all())))
V = pd.DataFrame(V)
V.to_csv(f"{OUT}/fase11_veredicto.csv", index=False)
M = u.groupby(["dataset", "kind"])[["acc", "fid", "nodes"]].mean()

# ------------------------------------------------------------------ B. consultas efetivas
def counts(root, tag):
    out = []
    for f in glob.glob(f"{root}/audits/*__fase11_oraculo.jsonl"):
        ds, s = os.path.basename(f).split("__")[0].rsplit("_seed", 1)
        for line in open(f):
            a = json.loads(line)
            if not (a["arm"] or "").startswith("sc_"):
                continue
            out.append(dict(execucao=tag, dataset=ds, seed=int(s), kind=a["arm"].split("_")[1], novos=a["unique"]["pontos_novos_rotulados"],
                            novos_proba=a["unique"]["pontos_novos_so_probabilidade"], contadas=sum(a["rows"].values())))
    return pd.DataFrame(out)


Q = pd.concat([counts(SEL, "selecionado"), counts(OMI, "omissao")], ignore_index=True)
Q.to_csv(f"{OUT}/fase11_contagens_por_arvore.csv", index=False)
QU = Q.groupby(["execucao", "dataset", "seed", "kind"])[["novos", "novos_proba", "contadas"]].mean().reset_index()
hq = []
for (ex, ds, s), g in QU.groupby(["execucao", "dataset", "seed"]):
    g = g.set_index("kind")
    for k, bk in (("E", "B"), ("D", "B"), ("A", "B"), ("V", "B"), ("R", "B"), ("R7", "B7")):
        if k in g.index and bk in g.index:
            hq.append(dict(execucao=ex, dataset=ds, seed=s, braco=k, razao_novos=g.loc[k, "novos"] / g.loc[bk, "novos"],
                           razao_contadas=g.loc[k, "contadas"] / g.loc[bk, "contadas"], novos_so_proba=g.loc[k, "novos_proba"],
                           ok=bool(g.loc[k, "novos"] <= 1.10 * g.loc[bk, "novos"] and g.loc[k, "novos_proba"] == 0)))
HQ = pd.DataFrame(hq)
HQ.to_csv(f"{OUT}/fase11_consultas_por_unidade.csv", index=False)
eqv = []
for ds in ["german_credit", "phys_bmi", "phys_kinetic", "phys_power"]:
    for s in SEEDS:
        for k in ("R", "R7"):
            for r in range(3):
                a, b = pred(OMI, ds, s, f"sc_{k}_r{r}"), pred(OMI, ds, s, f"sc_{k}np_r{r}")
                eqv.append(dict(dataset=ds, seed=s, braco=k, rep=r, igual=a is not None and b is not None and a["predictions"] == b["predictions"]))
EQ = pd.DataFrame(eqv)
EQ.to_csv(f"{OUT}/fase11_equivalencia_Rnp_omissao.csv", index=False)

# ------------------------------------------------------------------ tabelas
fmt = lambda r: f"{r.media * 100:+.2f} [{r.lo * 100:+.2f}; {r.hi * 100:+.2f}]".replace(".", ",")
def tab(lab):
    s = S[S.orcamento == lab]
    out = ["| Contraste (pp) | " + " | ".join(DS) + " |", "|---" * (len(DS) + 1) + "|"]
    for c in s.contraste.unique():
        out.append("| " + c + " | " + " | ".join(fmt(s[(s.contraste == c) & (s.dataset == d)].iloc[0]) for d in DS) + " |")
    return "\n".join(out)
f_ = lambda v, n=3: f"{v:.{n}f}".replace(".", ",")
NT = ["| Dataset | Nós C4.5 | N (limite de Bc/Rc) | Nós Bc | Nós Rc | Nós B | Nós R | Nós c45_d2 | Nós B7 | Nós R7 |", "|---|---|---|---|---|---|---|---|---|---|"]
for ds in DS:
    k = K[K.dataset == ds]
    NT.append(f"| {ds} | {f_(M.loc[(ds, 'C4.5'), 'nodes'], 1)} | {f_(k.N.mean(), 1)} | {f_(M.loc[(ds, 'Bc'), 'nodes'], 1)} | {f_(M.loc[(ds, 'Rc'), 'nodes'], 1)} | "
              f"{f_(M.loc[(ds, 'B'), 'nodes'], 1)} | {f_(M.loc[(ds, 'R'), 'nodes'], 1)} | {f_(M.loc[(ds, 'C4.5d2'), 'nodes'], 1)} | {f_(M.loc[(ds, 'B7'), 'nodes'], 1)} | {f_(M.loc[(ds, 'R7'), 'nodes'], 1)} |")
VT = ["| Orçamento | Dataset | Elegível | Hierarquia sustentada | Hierarquia observada | Original > C4.5 sustentado | Reloaded > C4.5 em acc. e fid. sustentado |", "|---|---|---|---|---|---|---|"]
for r in V.itertuples():
    VT.append(f"| {r.orcamento} | {r.dataset} | {'sim' if r.elegivel else 'não'} | {'sim' if r.sustentada else 'não'} | {'sim' if r.observada else 'não'} | "
              f"{'sim' if r.original_supera_c45 else 'não'} | {'sim' if r.reloaded_supera_c45_acc_e_fid else 'não'} |")
QT = ["| Execução | Dataset | Braço | Razão pontos novos (mín.–máx.) | Razão linhas contadas (mín.–máx.) | Unidades com H11.1/H11.2 | Pontos novos só por probabilidade |", "|---|---|---|---|---|---|---|"]
for (ex, ds, k), g in HQ.groupby(["execucao", "dataset", "braco"]):
    QT.append(f"| {ex} | {ds} | {k} | {f_(g.razao_novos.min(), 2)}–{f_(g.razao_novos.max(), 2)} | {f_(g.razao_contadas.min(), 2)}–{f_(g.razao_contadas.max(), 2)} | "
              f"{int(g.ok.sum())}/{len(g)} | {int(g.novos_so_proba.sum())} |")
txt = "\n@@\n".join([tab("tamanho do C4.5"), tab("7 nós"), "\n".join(NT), "\n".join(VT), "\n".join(QT)])
txt += (f"\n@@\nIntegridade: {int(I.igual.sum())}/{len(I)}; unidades com diferenças: {sorted(bad)}\n"
        f"Construção: Bc {int(K.Bc_ok.sum())}/{len(K)}, Rc {int(K.Rc_ok.sum())}/{len(K)}, c45_d2 {int(K.d2_ok.sum())}/{len(K)}\n"
        f"Rnp ≡ R (MLP por omissão): {int(EQ.igual.sum())}/{len(EQ)}")
open(f"{OUT}/fase11_tabelas.txt", "w").write(txt)
print(txt)
