"""Ponto 1 da fase 7: o MLP original memoriza os rótulos trocados (ruído injetado), ou apenas sobreajusta?
Reconstrói, com a semente fixa do gerador, a máscara de rótulos trocados e o rótulo limpo de cada linha. Reajusta o MLP original (fit_mlp,
trials=0) e o otimizado (select_and_fit) no treino de cada unidade e verifica que reproduzem as previsões de teste guardadas nas fases 6 e 7.
Análise POST HOC (não pré-registada) e descritiva; não influencia nenhuma escolha."""
import sys, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/kambulu89/biuri_trepan_reloaded")
from validation.benchmark import run as R  # noqa  PRIMEIRO: fixa o ambiente numérico (como nas campanhas)
import numpy as np, pandas as pd
from validation.benchmark import datasets_physics as P
from validation.benchmark import datasets as ds_mod
from core.benchmark.runner import fit_mlp
from validation.benchmark.mlp_selection_fase7 import select_and_fit
from core.c45_j48_tree import C45Classifier
rows = []
for name in P.GENERATORS:
    rng = np.random.default_rng([P.GENERATION_SEED, list(P.GENERATORS).index(name)])
    X, y_clean, *_ = P.GENERATORS[name](rng)
    flipped = rng.random(len(y_clean)) < P.LABEL_NOISE
    ds, _, _ = ds_mod.REGISTRY[name].load()
    assert np.array_equal(ds.y, np.where(flipped, 1 - y_clean, y_clean)), "reconstrução falhou"
    for seed in (42, 7, 123, 2024, 11):
        j6 = json.load(open(f"/home/user/audit_runs/fase6/raw/{name}/seed{seed}/mlp_original.json"))["predictions"]
        j7 = json.load(open(f"/home/user/audit_runs/fase7/raw/{name}/seed{seed}/mlp_original.json"))["predictions"]
        te = np.asarray(j6["test_row_index"]); tr = np.setdiff1d(np.arange(len(ds.y)), te)
        models = {"MLP original": fit_mlp(ds.X[tr], ds.y[tr], seed, 0, 3)[0], "MLP otimizado": select_and_fit(ds.X[tr], ds.y[tr], seed)[0],
                  "C4.5": C45Classifier(confidence_factor=0.25, min_samples_leaf=2, random_state=seed).fit(ds.X[tr], ds.y[tr])}
        repro = {"MLP original": bool(np.array_equal(models["MLP original"].predict(ds.X[te]), np.asarray(j6["surrogate_predictions"]))),
                 "MLP otimizado": bool(np.array_equal(models["MLP otimizado"].predict(ds.X[te]), np.asarray(j7["surrogate_predictions"]))), "C4.5": None}
        f_tr, f_te = flipped[tr], flipped[te]
        for k, mdl in models.items():
            ptr, pte = mdl.predict(ds.X[tr]), mdl.predict(ds.X[te])
            rows.append(dict(dataset=name, seed=seed, modelo=k, reproduz=repro[k], n_trocados_treino=int(f_tr.sum()),
                             memorizacao=float(np.mean(ptr[f_tr] == ds.y[tr][f_tr])),          # prevê o rótulo TROCADO nas linhas trocadas
                             acc_treino_linhas_limpas=float(np.mean(ptr[~f_tr] == ds.y[tr][~f_tr])),
                             acc_treino_total=float(np.mean(ptr == ds.y[tr])),
                             acc_teste_rotulos_observados=float(np.mean(pte == ds.y[te])),
                             acc_teste_rotulos_limpos=float(np.mean(pte == y_clean[te]))))
        print(name, seed, {r["modelo"]: (round(r["memorizacao"], 2), r["reproduz"]) for r in rows[-3:]}, flush=True)
d = pd.DataFrame(rows); d.to_csv("/home/user/audit_work/fase7/memorizacao.csv", index=False)
print(d.groupby(["dataset", "modelo"])[["memorizacao", "acc_treino_linhas_limpas", "acc_treino_total", "acc_teste_rotulos_observados", "acc_teste_rotulos_limpos"]].mean().round(3).to_string())
print("reprodução:", d.groupby("modelo").reproduz.apply(lambda s: s.dropna().mean() if s.notna().any() else None).to_dict())
