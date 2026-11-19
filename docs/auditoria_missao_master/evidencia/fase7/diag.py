"""Diagnóstico do MLP (fase 7), SÓ com o treino de cada unidade da fase 6. O teste nunca é lido (apenas os índices, para obter o complemento)."""
import sys, json, warnings, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/kambulu89/biuri_trepan_reloaded")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from validation.benchmark import run as R  # noqa
from validation.benchmark import datasets as ds_mod
from validation.benchmark import datasets_physics, datasets_categorical  # noqa
from core.mlp_factory import build_mlp_for_data
from core.mlp_convergence import extract_mlp_convergence
from core.c45_j48_tree import C45Classifier
F6 = "/home/user/audit_runs/fase6/raw"
rows = []
for name in ("phys_bmi", "phys_kinetic", "phys_power", "german_credit"):
    ds, _, _ = ds_mod.REGISTRY[name].load()
    for seed in (42, 7, 123, 2024, 11):
        te = json.load(open(f"{F6}/{name}/seed{seed}/mlp_original.json"))["predictions"]["test_row_index"]
        tr = np.setdiff1d(np.arange(len(ds.y)), te); X, y = ds.X[tr], ds.y[tr]
        variants = {"default": {}, "alpha=1e-2": dict(alpha=1e-2), "alpha=1e-1": dict(alpha=1e-1), "alpha=1": dict(alpha=1.0),
                    "adam+early_stopping": dict(solver="adam", early_stopping=True, validation_fraction=0.2, learning_rate_init=1e-3, n_iter_no_change=30, max_iter=3000)}
        cv = StratifiedKFold(5, shuffle=True, random_state=seed)
        res = {k: [] for k in list(variants) + ["c45"]}
        for a, b in cv.split(X, y):
            for k, ov in variants.items():
                m = build_mlp_for_data(X[a], y[a], random_state=seed, **ov).fit(X[a], y[a])
                conv = extract_mlp_convergence(m)
                res[k].append((float(np.mean(m.predict(X[b]) == y[b])), float(np.mean(m.predict(X[a]) == y[a])), conv.get("converged"), conv.get("n_iter")))
            c = C45Classifier(confidence_factor=0.25, min_samples_leaf=2, random_state=seed).fit(X[a], y[a])
            res["c45"].append((float(np.mean(c.predict(X[b]) == y[b])), float(np.mean(c.predict(X[a]) == y[a])), None, None))
        prof = build_mlp_for_data(X, y, random_state=seed).BIURI_ADAPTIVE_PROFILE["params"]
        for k, v in res.items():
            v = np.array(v, dtype=object)
            rows.append(dict(dataset=name, seed=seed, variant=k, cv_acc=float(np.mean(v[:, 0].astype(float))), train_acc=float(np.mean(v[:, 1].astype(float))),
                             converged_frac=(float(np.mean([bool(x) for x in v[:, 2]])) if k != "c45" else None),
                             n_iter_mean=(float(np.mean([x for x in v[:, 3] if x is not None])) if k != "c45" else None)))
        print(name, seed, {k: round(float(np.mean(np.array(v, dtype=object)[:, 0].astype(float))), 3) for k, v in res.items()}, flush=True)
    print(name, "perfil por omissão:", {k: prof[k] for k in ("hidden_layer_sizes", "solver", "alpha", "max_iter", "early_stopping")}, flush=True)
d = pd.DataFrame(rows); d.to_csv("/home/user/audit_work/fase7/diag.csv", index=False)
print(d.groupby(["dataset", "variant"])[["cv_acc", "train_acc", "converged_frac", "n_iter_mean"]].mean().round(3).to_string())
