"""B3-Q / Doc. 57: CALIBRAÇÃO INDEPENDENTE para a proposta de limiares Δ_o (aprendibilidade do oráculo), τ_s (identificação supervisionada genérica) e τ_u (recuperabilidade não supervisionada).
Família aditiva por conceito (R8, exploratória), sementes NOVAS (base 9500; disjuntas de 9191 e de todas as anteriores), 30 réplicas por célula. Só gerador + MLP + métodos genéricos; sem ontologia,
sem TREPAN, sem braços OWL, sem teste da avaliação. Guarda, por célula, as distribuições (não só medianas) e os erros-padrão, para fundamentar as margens."""
import json
import sys
import warnings

import numpy as np
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main(out, reps=30, K=8, g=6):
    warnings.filterwarnings("ignore")
    p = K * g
    true = np.repeat(np.arange(K), g)
    res = {}
    for s in (1.0, 2.0, 3.0, 4.0):
        for n in (60, 120, 240, 1000):
            D, hit, ari, acc = [], [], [], []
            for r in range(reps):
                rng = np.random.default_rng([9500, n, int(s * 10), r])
                A = rng.choice(K, 1)[0]
                idx = np.arange(A * g, (A + 1) * g)

                def draw(m):
                    X = rng.normal(size=(m, p))
                    return X, (X[:, idx].sum(1) + s * rng.normal(size=m) > 0).astype(int)
                Xtr, ytr = draw(n)
                Xva, yva = draw(2000)
                mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=r)).fit(Xtr, ytr)
                a = float(np.mean(mlp.predict(Xva) == yva))
                acc.append(a)
                D.append(a - float(max(yva.mean(), 1 - yva.mean())))
                c = np.abs([np.corrcoef(Xtr[:, j], ytr)[0, 1] for j in range(p)])
                h1 = len(set(np.argsort(-c)[:g]) & set(idx)) / g
                lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.3).fit(StandardScaler().fit_transform(Xtr), ytr)
                h2 = len(set(np.argsort(-np.abs(lr.coef_[0]))[:g]) & set(idx)) / g
                hit.append(max(h1, h2))
                Aff = np.abs(np.nan_to_num(np.corrcoef(Xtr, rowvar=False)))
                np.fill_diagonal(Aff, 0.0)
                lab = SpectralClustering(K, affinity="precomputed", random_state=0).fit_predict(Aff + 1e-9)
                ari.append(adjusted_rand_score(true, lab))
            D, hit, ari = np.array(D), np.array(hit), np.array(ari)
            se = lambda v: float(np.std(v, ddof=1) / np.sqrt(len(v)))
            res[f"s{s}_n{n}"] = {"Do_mean": float(D.mean()), "Do_sd": float(D.std(ddof=1)), "Do_se": se(D), "Do_p10": float(np.quantile(D, .1)), "Do_min": float(D.min()), "frac_Do_gt0": float((D > 0).mean()),
                                 "hit_mean": float(hit.mean()), "hit_se": se(hit), "hit_p90": float(np.quantile(hit, .9)), "ari_median": float(np.median(ari)), "ari_p90": float(np.quantile(ari, .9)),
                                 "mlp_acc_mean": float(np.mean(acc))}
            print(f"s={s} n={n}: Do mean {D.mean():.3f} sd {D.std(ddof=1):.3f} p10 {np.quantile(D,.1):.3f} | hit mean {hit.mean():.2f} p90 {np.quantile(hit,.9):.2f} | ARI med {np.median(ari):.2f}", flush=True)
    json.dump({"reps": reps, "seed_base": 9500, "K": K, "g": g, "labels_used": "só treino/validação do gerador; sem ontologia/TREPAN/braços OWL/teste da avaliação", "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
