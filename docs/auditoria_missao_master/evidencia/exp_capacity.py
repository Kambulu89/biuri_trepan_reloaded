"""Exploratório (DEV): fidelidade/accuracy do TREPAN Original e de um C4.5 treinado com rótulos do oráculo em função do nº de nós,
com o mesmo MLP e a mesma partição. Só usa parâmetros já existentes; não altera o repositório.
Variantes do Original: 12 atributos por nó (omissão) e todos os atributos (WIDE)."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/kambulu89/biuri_trepan_reloaded")
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import train_test_split
from core.mlp_factory import build_mlp_for_data
from core.trepan_original import TrepanOriginalClassifier
from core.c45_j48_tree import C45Classifier
rows = []
for name, loader in (("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer), ("digits", load_digits)):
    d = loader(); X, y = d.data.astype(float), d.target
    for seed in (42, 7, 123, 2024, 11):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        mlp = build_mlp_for_data(Xtr, ytr, random_state=seed).fit(Xtr, ytr)
        ote, otr = mlp.predict(Xte), mlp.predict(Xtr)
        base = dict(dataset=name, seed=seed, mlp_acc=float(np.mean(ote == yte)))
        for label, ytrain in (("c45_real_labels", ytr), ("c45_oracle_labels", otr)):
            c = C45Classifier().fit(Xtr, ytrain); p = c.predict(Xte)
            rows.append(dict(base, arm=label, max_nodes=None, nodes=int(2 * c.get_n_leaves() - 1) if True else None, leaves=int(c.get_n_leaves()),
                             acc=float(np.mean(p == yte)), fid=float(np.mean(p == ote))))
            print(json.dumps(rows[-1]), flush=True)
        n = len(Xtr); ms = max(n, min(1000, max(120, 3 * n)))
        for mn in (7, 15, 31, 63, 127):
            for label, k in (("original_k12", 12), ("original_wide", X.shape[1])):
                if name == "iris" and label == "original_wide": continue          # 4 atributos: idêntico a k12
                t0 = time.perf_counter()
                t = TrepanOriginalClassifier(random_state=seed, max_features_per_node=min(k, X.shape[1]), min_sample=ms, max_queries=mn * ms,
                                             max_nodes=mn, max_depth=31, max_n=3, beam_width=2, min_samples_leaf=2).fit(Xtr, oracle=mlp)
                p = t.predict(Xte)
                rows.append(dict(base, arm=label, max_nodes=mn, nodes=int(t.node_count_), leaves=int(t.get_n_leaves()),
                                 acc=float(np.mean(p == yte)), fid=float(np.mean(p == ote)), queries=int(t.membership_queries_),
                                 secs=round(time.perf_counter() - t0, 1)))
                print(json.dumps(rows[-1]), flush=True)
        json.dump(rows, open("/home/user/audit_logs/exp_capacity.json", "w"), indent=1)
