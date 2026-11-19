"""Variante identificada separadamente: C4.5 treinado com os rótulos do ORÁCULO (não é o baseline B).
Usa as mesmas partições e o mesmo MLP das unidades do benchmark (verifica que o MLP reajustado reproduz as previsões guardadas)."""
import sys, json, glob, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/kambulu89/biuri_trepan_reloaded")
import numpy as np
from validation.benchmark import datasets as ds_mod
from core.benchmark.runner import fit_mlp
from core.c45_j48_tree import C45Classifier
root = sys.argv[1]; out = []
for u in sorted(glob.glob(root + "/raw/*/seed*/")):
    m = json.load(open(u + "mlp_original.json")); name, seed = m["dataset"], int(m["master_seed"])
    ds, _, _ = ds_mod.REGISTRY[name].load()
    te = np.array(m["predictions"]["test_row_index"]); tr = np.setdiff1d(np.arange(len(ds.y)), te)
    stored = np.array(m["predictions"]["surrogate_predictions"])
    mlp, _ = fit_mlp(ds.X[tr], ds.y[tr], seed, 0, 3)
    ote = np.asarray(mlp.predict(ds.X[te]))
    same = bool((ote == stored).all())
    c = C45Classifier().fit(ds.X[tr], np.asarray(mlp.predict(ds.X[tr])))
    p = c.predict(ds.X[te]); yt = np.array(m["predictions"]["y_true"])
    row = dict(dataset=name, seed=seed, arm="c45_oracle_labels", mlp_reproduced=same, accuracy_real_labels=float(np.mean(p == yt)),
               fidelity_to_oracle=float(np.mean(p == stored)), leaf_count=int(c.get_n_leaves()), depth=int(c.get_depth()))
    out.append(row); print(json.dumps(row), flush=True)
json.dump(out, open(sys.argv[2], "w"), indent=1)
