"""Ponto 3: a melhoria do MLP chega às árvores? Accuracy (rótulos reais) de cada árvore com o oráculo da fase 7 menos com o da fase 6,
emparelhada por unidade (mesma partição de teste). Unidade = semente; as 3 réplicas são médias DENTRO da unidade antes do bootstrap."""
import json, glob, os, sys, numpy as np, pandas as pd
def load(root):
    rows = []
    for f in sorted(glob.glob(root + "/raw/*/seed*/*.json")):
        b = os.path.basename(f)[:-5]
        if b in ("split", "UNIT_SHA256"): continue
        r = json.load(open(f))["row"]
        k = b.split("_")[1] if b.startswith("sc_") else {"mlp_original": "MLP", "c45": "C4.5"}.get(b)
        if k: rows.append(dict(dataset=r["dataset"], seed=int(r["seed"]), kind=k, acc=r["accuracy_real_labels"], f1=r.get("macro_f1_real_labels")))
    return pd.DataFrame(rows).groupby(["dataset", "seed", "kind"]).mean().reset_index()
a6, a7 = load("/home/user/audit_runs/fase6"), load("/home/user/audit_runs/fase7")
m = a6.merge(a7, on=["dataset", "seed", "kind"], suffixes=("_f6", "_f7")); m["d_acc"] = m.acc_f7 - m.acc_f6
rng = np.random.default_rng(20261123); out = []
for (ds, k), g in m.groupby(["dataset", "kind"]):
    x = g.sort_values("seed").d_acc.to_numpy(); b = rng.choice(x, (20000, len(x))).mean(1)
    out.append(dict(dataset=ds, braco=k, delta_acc=round(x.mean(), 4), lo=round(np.quantile(b, .025), 4), hi=round(np.quantile(b, .975), 4)))
o = pd.DataFrame(out); mlp = o[o.braco == "MLP"].set_index("dataset").delta_acc
o["fracao_do_ganho_do_MLP"] = o.apply(lambda r: round(r.delta_acc / mlp[r.dataset], 2) if abs(mlp[r.dataset]) > 1e-9 else None, axis=1)
o.to_csv(sys.argv[1], index=False); print(o[o.braco.isin(["MLP", "C4.5", "B", "R", "Rq", "B7", "R7", "Rq7"])].to_string())
