"""B3-Q / Doc. 49 §3 (ponto 7): o MLP aprende a tarefa do gerador LATENT_CONCEPT_V1 quando a partição NÃO é recuperável? Três famílias de tarefa sobre conceitos latentes (q=0,65, η=0,05):
``single`` y = 1[z_A > t]; ``and`` y = 1[z_A>t] ∧ 1[z_B>t]; ``xor`` y = 1[z_A>t] ⊕ 1[z_B>t]. Só gerador + MLP (sem TREPAN, sem ontologia, sem braços OWL, sem teste da avaliação): sementes de
CALIBRAÇÃO e validação gerada de novo (2000 exemplos). Reporta a accuracy do MLP, a taxa da classe maioritária e o intervalo 10–90 % entre réplicas."""
import json
import sys
import warnings

import numpy as np
from scipy.stats import norm
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

TASKS = ("single", "and", "xor")


def label(z, A, B, task, t):
    a, b = z[:, A] > t, z[:, B] > t
    return (a if task == "single" else (a & b) if task == "and" else (a ^ b)).astype(int)


def main(out, reps=15, K=8, g=6, eta=0.05, q=0.65):
    warnings.filterwarnings("ignore")
    res = {}
    t = norm.ppf(q)
    for task in TASKS:
        for (lo, hi) in ((0.60, 0.85), (0.45, 0.65), (0.35, 0.55), (0.25, 0.45)):
            for n in (60, 120, 240, 1000):
                acc, maj = [], []
                for r in range(reps):
                    A, B = np.random.default_rng([8181, r]).choice(K, size=2, replace=False)
                    lam = np.random.default_rng([8181, n, int(lo * 100), r, 1]).uniform(lo, hi, size=(K, g))

                    def draw(m, rr):
                        z = rr.normal(size=(m, K))
                        X = np.empty((m, K * g))
                        for c in range(K):
                            for j in range(g):
                                X[:, c * g + j] = lam[c, j] * z[:, c] + np.sqrt(1 - lam[c, j] ** 2) * rr.normal(size=m)
                        y = label(z, A, B, task, t)
                        return X, np.where(rr.random(m) < eta, 1 - y, y)
                    Xtr, ytr = draw(n, np.random.default_rng([8181, n, int(lo * 100), r, 2]))
                    Xva, yva = draw(2000, np.random.default_rng([8181, n, int(lo * 100), r, 3]))
                    mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=r)).fit(Xtr, ytr)
                    acc.append(float(np.mean(mlp.predict(Xva) == yva)))
                    maj.append(float(max(yva.mean(), 1 - yva.mean())))
                res[f"{task}_lam{lo}-{hi}_n{n}"] = {"mlp_accuracy_median": float(np.median(acc)), "mlp_accuracy_p10": float(np.quantile(acc, .1)), "mlp_accuracy_p90": float(np.quantile(acc, .9)),
                                                    "majority_rate_median": float(np.median(maj))}
                print(f"{task} lam{lo}-{hi} n={n}: MLP {np.median(acc):.3f} (p10 {np.quantile(acc, .1):.3f}) vs maioria {np.median(maj):.3f}", flush=True)
    json.dump({"reps": reps, "K": K, "g": g, "eta": eta, "q": q, "labels_used": "só treino/validação do gerador em sementes de calibração; sem teste da avaliação; sem braços OWL", "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
