"""Exploratório (DEV): efeito do amostrador conjunto empírico já existente em core/query_sampling.py (desligado por omissão)
no TREPAN Original, face ao amostrador marginal e ao C4.5. Mesmo MLP e partição por unidade. Não altera o repositório."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/kambulu89/biuri_trepan_reloaded")
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import train_test_split
from core.mlp_factory import build_mlp_for_data
from core.trepan_original import TrepanOriginalClassifier
from core.c45_j48_tree import C45Classifier
from core.query_sampling import joint_sampler_context
rows = []
for name, loader in (("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer), ("digits", load_digits)):
    d = loader(); X, y = d.data.astype(float), d.target
    for seed in (42, 7, 123, 2024, 11):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        mlp = build_mlp_for_data(Xtr, ytr, random_state=seed).fit(Xtr, ytr); ote = mlp.predict(Xte)
        c = C45Classifier().fit(Xtr, ytr); pc = c.predict(Xte)
        base = dict(dataset=name, seed=seed, mlp_acc=float(np.mean(ote == yte)), c45_acc=float(np.mean(pc == yte)), c45_fid=float(np.mean(pc == ote)),
                    c45_nodes=int(2 * c.get_n_leaves() - 1))
        n = len(Xtr); ms = max(n, min(1000, max(120, 3 * n))); mn = 31
        for sampler in ("marginal", "joint"):
            for label, k in (("k12", 12), ("wide", X.shape[1])):
                if X.shape[1] <= 12 and label == "wide": continue
                for rep in range(3):
                    def fit():
                        return TrepanOriginalClassifier(random_state=seed + 1000 * rep, max_features_per_node=min(k, X.shape[1]), min_sample=ms,
                                                        max_queries=mn * ms, max_nodes=mn, max_depth=31, max_n=3, beam_width=2,
                                                        min_samples_leaf=2).fit(Xtr, oracle=mlp)
                    if sampler == "joint":
                        with joint_sampler_context(replica=rep):
                            t = fit()
                    else:
                        t = fit()
                    p = t.predict(Xte)
                    rows.append(dict(base, sampler=sampler, feats=label, replica=rep, nodes=int(t.node_count_), acc=float(np.mean(p == yte)),
                                     fid=float(np.mean(p == ote)), queries=int(t.membership_queries_)))
                    print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open("/home/user/audit_logs/exp_sampler.json", "w"), indent=1)
