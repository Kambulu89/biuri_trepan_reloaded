"""Auditoria: efeito da pré-seleção de atributos por variância bruta (max_features_per_node) no TREPAN Original.
Não altera o repositório. Mesmo MLP/oráculo, mesmas partições; só muda max_features_per_node ou a escala das colunas."""
import sys, json, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/kambulu89/biuri_trepan_reloaded")
import numpy as np
from sklearn.datasets import load_breast_cancer, load_wine, load_digits, load_iris
from sklearn.model_selection import train_test_split
from core.mlp_factory import build_mlp_for_data
from core.trepan_original import TrepanOriginalClassifier
from core.c45_j48_tree import C45Classifier

class Scaled:
    """Oráculo equivalente num espaço reescalado: f'(x') = f(x'/s). A função é a mesma."""
    def __init__(self, m, s): self.m, self.s = m, s
    def predict(self, X): return self.m.predict(np.asarray(X, float) / self.s)

def budget(n):
    ms = max(n, min(1000, max(120, 3 * n)))
    return dict(min_sample=ms, max_queries=31 * ms, max_nodes=31, max_depth=8, max_n=3, beam_width=2, min_samples_leaf=2)

rows = []
for name, loader in (("iris", load_iris), ("wine", load_wine), ("breast_cancer", load_breast_cancer), ("digits", load_digits)):
    d = loader(); X, y = d.data.astype(float), d.target
    for seed in (42, 7, 123, 2024, 11):
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
        mlp = build_mlp_for_data(Xtr, ytr, random_state=seed).fit(Xtr, ytr)
        ote = mlp.predict(Xte)
        c45 = C45Classifier().fit(Xtr, ytr)
        base = dict(dataset=name, seed=seed, n_features=X.shape[1], mlp_acc=float(np.mean(ote == yte)),
                    c45_acc=float(np.mean(c45.predict(Xte) == yte)), c45_fid=float(np.mean(c45.predict(Xte) == ote)))
        b = budget(len(Xtr))
        s = Xtr.std(axis=0); s[s < 1e-12] = 1.0
        variants = {"k12_raw": (12, None), "kall_raw": (X.shape[1], None), "k12_standardised": (12, 1.0 / s)}
        for vname, (k, scale) in variants.items():
            t0 = time.perf_counter()
            if scale is None:
                A, B, orc = Xtr, Xte, mlp
            else:
                A, B, orc = Xtr * scale, Xte * scale, Scaled(mlp, scale)
            t = TrepanOriginalClassifier(random_state=seed, max_features_per_node=min(k, X.shape[1]), **b).fit(A, oracle=orc)
            p = t.predict(B)
            rows.append(dict(base, variant=vname, acc=float(np.mean(p == yte)), fid=float(np.mean(p == ote)),
                             nodes=int(t.node_count_), secs=round(time.perf_counter() - t0, 1)))
            print(json.dumps(rows[-1]), flush=True)
json.dump(rows, open("/home/user/audit_logs/exp_variance_prefilter.json", "w"), indent=1)
