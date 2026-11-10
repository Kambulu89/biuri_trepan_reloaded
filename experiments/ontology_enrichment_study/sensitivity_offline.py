"""ANÁLISE DE SENSIBILIDADE (NÃO oficial; não toca no BIURI): em CV interna sobre o treino, compara um MLP com
(a) 48 colunas, (b) 48 + agregados corretos, (c) só os 8 agregados, (d) 48 + agregados permutados.
Serve só para diagnosticar se a informação semântica existe nas features derivadas (Objetivo 4)."""
import sys, numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
sys.path.insert(0, "."); import gen_synthetic as G
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1
X, y, names, groups = G.make_s1(seed)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)
idx = {n: i for i, n in enumerate(names)}
def agg(Xm, gr): return np.column_stack([Xm[:, [idx[c] for c in cs]].mean(1) for cs in gr.values()])
import random; pg = G.permuted(groups, random.Random(seed * 7919))
cv = StratifiedKFold(5, shuffle=True, random_state=0)
mk = lambda: make_pipeline(StandardScaler(), MLPClassifier((64, 32), max_iter=800, random_state=0, alpha=1e-3))
for lab, f in (("todas as colunas", lambda M: M), ("todas + agregados corretos", lambda M: np.hstack([M, agg(M, groups)])),
               ("só agregados corretos", lambda M: agg(M, groups)), ("todas + agregados permutados", lambda M: np.hstack([M, agg(M, pg)])),
               ("só agregados permutados", lambda M: agg(M, pg))):
    s = cross_val_score(mk(), f(Xtr), ytr, cv=cv).mean(); m = mk().fit(f(Xtr), ytr)
    print(f"seed{seed} {lab:32s} CV-treino={s:.3f}  teste={m.score(f(Xte), yte):.3f}")
