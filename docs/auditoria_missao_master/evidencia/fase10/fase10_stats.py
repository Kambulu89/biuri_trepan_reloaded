"""Análise da fase 10 (pré-registo: FASE10_PREREGISTO.md). Uso: python fase10_stats.py RAIZ_FASE10 OUT_DIR"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
PRIOR = {d: "/home/user/audit_runs/fase8" for d in ("iris", "wine", "breast_cancer", "digits", "adult")}
PRIOR.update({d: "/home/user/audit_runs/fase9" for d in ("german_credit", "phys_bmi", "phys_kinetic", "phys_power")})
DS = ["iris", "wine", "breast_cancer", "digits", "adult", "german_credit", "phys_bmi", "phys_kinetic", "phys_power"]

# --- contagens por árvore
rows = []
for f in sorted(glob.glob(f"{ROOT}/audits/*__fase10_oraculo.jsonl")):
    ds, s = os.path.basename(f).split("__")[0].rsplit("_seed", 1)
    for line in open(f):
        a = json.loads(line)
        arm = a["arm"] or ""
        if not arm.startswith("sc_"):
            continue
        kind, rep = arm.split("_")[1], int(arm.rsplit("_r", 1)[1])
        u = a["unique"]
        rows.append(dict(dataset=ds, seed=int(s), kind=kind, rep=rep, novos=u["pontos_novos_rotulados"], novos_proba=u["pontos_novos_so_probabilidade"],
                         reproba=u["reconsultas_probabilidade_de_pontos_conhecidos"], relabel=u["reconsultas_rotulo_de_pontos_conhecidos"],
                         linhas_contadas=sum(a["rows"].values()), chamadas_proba=a["calls"].get("predict_proba", 0)))
C = pd.DataFrame(rows)
C.to_csv(f"{OUT}/fase10_contagens_por_arvore.csv", index=False)
U = C.groupby(["dataset", "seed", "kind"])[["novos", "novos_proba", "reproba", "relabel", "linhas_contadas"]].mean().reset_index()

# --- previsões
def preds(root, ds, s, arm):
    p = f"{root}/raw/{ds}/seed{s}/{arm}.json"
    return json.load(open(p))["predictions"] if os.path.exists(p) else None

integ, equiv = [], []
for ds in DS:
    for s in (42, 7, 123, 2024, 11):
        if not os.path.exists(f"{ROOT}/raw/{ds}/seed{s}/RUN_META.txt"):
            continue
        for arm in ["mlp_original", "c45"] + [f"sc_{k}_r{r}" for k in ("B", "R", "B7", "R7") for r in range(3)]:
            a, b = preds(ROOT, ds, s, arm), preds(PRIOR[ds], ds, s, arm)
            integ.append(dict(dataset=ds, seed=s, arm=arm, igual=(a is not None and a == b)))
        for k in ("R", "R7"):
            for r in range(3):
                a, b = preds(ROOT, ds, s, f"sc_{k}_r{r}"), preds(ROOT, ds, s, f"sc_{k}np_r{r}")
                equiv.append(dict(dataset=ds, seed=s, orcamento="7 nós" if k == "R7" else "tuning", rep=r, igual=(a is not None and a == b)))
I, E = pd.DataFrame(integ), pd.DataFrame(equiv)
I.to_csv(f"{OUT}/fase10_integridade.csv", index=False)
E.to_csv(f"{OUT}/fase10_equivalencia_Rnp.csv", index=False)
bad_units = set(map(tuple, I[~I.igual][["dataset", "seed"]].drop_duplicates().values))

# --- H10.1 por unidade e orçamento
h = []
for (ds, s), g in U.groupby(["dataset", "seed"]):
    for lab, rk, bk in (("tuning", "R", "B"), ("7 nós", "R7", "B7")):
        r, b = g[g.kind == rk].iloc[0], g[g.kind == bk].iloc[0]
        h.append(dict(dataset=ds, seed=s, orcamento=lab, novos_B=b.novos, novos_R=r.novos, razao_novos=r.novos / b.novos,
                      novos_so_proba_R=r.novos_proba, reproba_R=r.reproba, contadas_B=b.linhas_contadas, contadas_R=r.linhas_contadas,
                      razao_contadas=r.linhas_contadas / b.linhas_contadas, H101=bool(r.novos <= 1.10 * b.novos and r.novos_proba == 0),
                      integridade=(ds, s) not in bad_units))
H = pd.DataFrame(h)
H.to_csv(f"{OUT}/fase10_por_unidade.csv", index=False)

f = lambda v, n=2: f"{v:.{n}f}".replace(".", ",")
lines = ["| Dataset | Orçamento | Pontos novos B (média) | Pontos novos R (média) | Razão novos R/B (mín.–máx.) | Razão linhas contadas R/B (mín.–máx.) | Reconsultas de probabilidade em R (média) | H10.1 (unidades) | H10.2: Rnp ≡ R (árvores) | Integridade (unidades) |",
         "|---|---|---|---|---|---|---|---|---|---|"]
summary = []
for ds in DS:
    for lab in ("tuning", "7 nós"):
        g = H[(H.dataset == ds) & (H.orcamento == lab)]
        if g.empty:
            continue
        e = E[(E.dataset == ds) & (E.orcamento == lab)]
        ok_units = int(g.integridade.sum())
        lines.append(f"| {ds} | {lab} | {g.novos_B.mean():.0f} | {g.novos_R.mean():.0f} | {f(g.razao_novos.min())}–{f(g.razao_novos.max())} | "
                     f"{f(g.razao_contadas.min())}–{f(g.razao_contadas.max())} | {g.reproba_R.mean():.0f} | {int(g.H101.sum())}/{len(g)} | "
                     f"{int(e.igual.sum())}/{len(e)} | {ok_units}/{len(g)} |")
        summary.append(dict(dataset=ds, orcamento=lab, H101_todas=bool(g.H101.all()), H102_todas=bool(e.igual.all()),
                            integridade_todas=bool(g.integridade.all()), decisao=bool(g.H101.all() and e.igual.all() and g.integridade.all())))
S = pd.DataFrame(summary)
S.to_csv(f"{OUT}/fase10_decisao.csv", index=False)
txt = "\n".join(lines)
txt += "\n\nIntegridade: " + f"{int(I.igual.sum())}/{len(I)} previsões iguais às fases 8/9; unidades com diferenças: {sorted(bad_units)}"
txt += "\nH10.2: " + f"{int(E.igual.sum())}/{len(E)} árvores Rnp/R7np iguais a R/R7"
txt += "\nDecisão (H10.1 e H10.2 e integridade em todas as unidades):\n" + S.to_string(index=False)
open(f"{OUT}/fase10_tabelas.txt", "w").write(txt)
print(txt)
