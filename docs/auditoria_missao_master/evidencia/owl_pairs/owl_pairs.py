"""Pares «com OWL − sem OWL» a partir dos resultados brutos JÁ EXISTENTES das campanhas da auditoria (nenhuma execução nova).

Uso:  python owl_pairs.py RAIZ_DAS_CAMPANHAS SAIDA.csv
      (RAIZ_DAS_CAMPANHAS contém baseline_main/, fase2_sampler/, ..., fase9/, cada uma com raw/<dataset>/seed<k>/<braço>.json)

Para cada par (campanha, braço com OWL, braço sem OWL) e dataset:
- diferença média por semente (réplicas promediadas dentro da semente), em pontos percentuais, de accuracy face aos rótulos reais e de
  fidelidade face ao MLP da MESMA unidade; intervalo bootstrap percentil a 95 % sobre as sementes (20 000 reamostragens, semente 1);
- verificação, semente a semente, de que os dois braços usam as mesmas linhas de teste, o mesmo oráculo congelado (``oracle_id``; ``fidelity_oracle_id`` vem vazio nos braços de árvore) e o
  mesmo orçamento CONFIGURADO (``tree_max_nodes``, ``query_budget``, ``tree_max_n``);
- consultas ao oráculo EFETIVAMENTE realizadas (média por árvore) em cada braço e a razão entre elas;
- o MLP usado como oráculo (``mlp_selection`` do braço ``mlp_original``: perfil por omissão ou seleção só no treino);
- estado: ``sustentado_ambas`` (os dois intervalos acima de 0), ``observado_ambas`` (as duas médias > 0, algum intervalo inclui 0),
  ``perda_sustentada`` (algum intervalo abaixo de 0), ``empate`` (as duas médias = 0) ou ``misto``.
Pares repetidos entre campanhas (mesmos resultados) aparecem uma vez, com todas as campanhas em ``campanhas``.
"""
import glob
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT, OUT = sys.argv[1], sys.argv[2]
CAMP = ["baseline_main", "fase2_sampler", "fase3_sampler", "fase4", "fase5", "fase6", "fase7", "fase8", "fase9"]
H = "Reloaded completo sem OWL (reloaded_core: mesmos mecanismos, sem ontologia)"
B = "TREPAN Original (B; equivalente ao Reloaded sem OWL pelo contrato EQ)"
B7 = "TREPAN Original com máx. 7 nós (B7)"
PAIRS = [(c, "reloaded_owl_full", "reloaded_core", "Reloaded completo com OWL, amostragem histórica", H) for c in CAMP]
PAIRS += [("fase2_sampler", "C", "E", "Reloaded completo com OWL, amostragem histórica, réplicas", "Reloaded completo sem OWL, histórico (E)"),
          ("fase2_sampler", "D", "F", "Reloaded completo com OWL + amostrador conjunto", "Reloaded completo sem OWL + amostrador conjunto (F)"),
          ("fase3_sampler", "D", "F", "Reloaded completo com OWL v2 + amostrador conjunto", "Reloaded completo sem OWL + amostrador conjunto (F)"),
          ("fase4", "O", "B", "O: atributos da OWL v2, sem mecanismos", B)]
DESC5 = {"O": "O: atributos OWL, sem mecanismos", "W": "W: canal de score", "E": "E: score + EFSR original", "A": "A: score + consultas ativas",
         "D": "D: Reloaded completo", "T": "T: desempate", "L": "L: limitada"}
PAIRS += [("fase5", k, "B", DESC5[k], B) for k in DESC5]
for c in ("fase6", "fase7", "fase8"):
    PAIRS += [(c, "R", "B", "R (= O): atributos OWL, sem mecanismos", B), (c, "R7", "B7", "R7: atributos OWL, máx. 7 nós", B7)]
DESC9 = {"R": "R (= O): atributos OWL", "E": "E: score + EFSR original", "V": "V: score + EFSR com lote novo"}
for k in DESC9:
    PAIRS += [("fase9", k, "B", DESC9[k], B), ("fase9", k + "7", "B7", DESC9[k] + ", máx. 7 nós", B7)]

rows, mlp_sel = [], {}
for c in CAMP:
    for f in glob.glob(f"{ROOT}/{c}/raw/*/seed*/*.json"):
        b = os.path.basename(f)[:-5]
        if b in ("split", "UNIT_SHA256"):
            continue
        j = json.load(open(f))
        r = j["row"]
        if b == "mlp_original":
            mlp_sel[(c, r["dataset"], int(r["seed"]))] = r.get("mlp_selection")
        kind = b.split("_")[1] if b.startswith("sc_") else b
        rows.append(dict(camp=c, ds=r["dataset"], seed=int(r["seed"]), kind=kind, acc=r["accuracy_real_labels"], fid=r.get("fidelity_to_oracle"),
                         oracle_id=r.get("fidelity_oracle_id") or r.get("oracle_id"), onto=r.get("ontology_hash"), queries=r.get("oracle_queries"),
                         test=hashlib.md5(json.dumps(j["predictions"].get("test_row_index"), default=str).encode()).hexdigest()[:12],
                         budget=json.dumps({k: r.get(k) for k in ("tree_max_nodes", "query_budget", "tree_max_n")}, default=str)))
D = pd.DataFrame(rows)
rng = np.random.default_rng(1)


def ci(x):
    x = np.asarray(x, float)
    bs = rng.choice(x, (20000, len(x))).mean(1)
    return float(np.quantile(bs, .025)), float(np.quantile(bs, .975))


out = []
for c, w, wo, desc, comp in PAIRS:
    d = D[D.camp == c]
    for ds in sorted(d.ds.unique()):
        a, b = d[(d.ds == ds) & (d.kind == w)], d[(d.ds == ds) & (d.kind == wo)]
        if a.empty or b.empty:
            continue
        ua, ub = a.groupby("seed")[["acc", "fid"]].mean(), b.groupby("seed")[["acc", "fid"]].mean()
        diff = (ua - ub).dropna() * 100
        seeds = list(diff.index)
        same = lambda col: all(set(a[a.seed == s][col]) == set(b[b.seed == s][col]) and a[a.seed == s][col].nunique() == 1 for s in seeds)
        la, ha = ci(diff.acc)
        lf, hf = ci(diff.fid)
        ma, mf = diff.acc.mean(), diff.fid.mean()
        if la > 0 and lf > 0:
            estado = "sustentado_ambas"
        elif ha < 0 or hf < 0:
            estado = "perda_sustentada"
        elif abs(ma) < 1e-9 and abs(mf) < 1e-9:
            estado = "empate"
        elif ma > 0 and mf > 0:
            estado = "observado_ambas"
        else:
            estado = "misto"
        sel = {mlp_sel.get((c, ds, s)) for s in seeds}
        out.append(dict(campanha=c, dataset=ds, com_owl=w, sem_owl=wo, variante=desc, comparador=comp,
                        oraculo="MLP por omissão (adaptive_factory_default)" if sel == {"adaptive_factory_default"} else
                        ("MLP selecionado só no treino (fase7_inner_cv_balanced_accuracy)" if sel == {"fase7_inner_cv_balanced_accuracy"} else str(sorted(map(str, sel)))),
                        sementes=len(seeds), replicas=int(a.groupby("seed").size().max()), ontologia=str(a.onto.iloc[0])[:8],
                        d_acc_pp=round(ma, 3), acc_ic_inf=round(la, 3), acc_ic_sup=round(ha, 3),
                        acc_sementes_pos=int((diff.acc > 1e-9).sum()), acc_sementes_neg=int((diff.acc < -1e-9).sum()),
                        d_fid_pp=round(mf, 3), fid_ic_inf=round(lf, 3), fid_ic_sup=round(hf, 3),
                        fid_sementes_pos=int((diff.fid > 1e-9).sum()), fid_sementes_neg=int((diff.fid < -1e-9).sum()),
                        mesmas_linhas_teste=same("test"), mesmo_oraculo=same("oracle_id"), mesmo_orcamento_configurado=same("budget"),
                        consultas_com_owl=round(float(a.queries.mean()), 1), consultas_sem_owl=round(float(b.queries.mean()), 1),
                        razao_consultas=round(float(a.queries.mean() / b.queries.mean()), 3), estado=estado))
O = pd.DataFrame(out)
key = ["dataset", "ontologia", "oraculo", "com_owl", "sem_owl", "d_acc_pp", "d_fid_pp", "consultas_com_owl"]
O["campanhas"] = O.groupby(key)["campanha"].transform(lambda s: ",".join(s))
O = O.drop_duplicates(key).drop(columns=["campanha"])
O = O[["campanhas"] + [c for c in O.columns if c != "campanhas"]]
O.to_csv(OUT, index=False)
print(len(O), "comparações distintas")
print(O.estado.value_counts().to_string())
