"""B3-Q / Doc. 49 §3 (ponto 7, alternativa): gerador ADITIVO de conceito — y = 1[ Σ_{j∈A} x_j + s·ε > 0 ] com indicadores x_j independentes N(0,1) e um conceito-alvo A de g indicadores (a regra é um
limiar de contagem ponderada, i.e. o alvo natural de um m-of-n sobre os indicadores do conceito). Como os x_j são independentes, a partição NÃO é recuperável de X por construção (recuperabilidade
não supervisionada nula); o que se mede é a identificabilidade SUPERVISIONADA (os g indicadores verdadeiros dentro dos g melhores por |corr| e por regressão logística L1) e a APRENDIBILIDADE
do MLP (accuracy vs classe maioritária). Só gerador + MLP + métodos genéricos, sementes de CALIBRAÇÃO, nenhuma ontologia/TREPAN/braço OWL, nenhum teste da avaliação."""
import json
import sys
import warnings

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main(out, reps=20, K=8, g=6):
    warnings.filterwarnings("ignore")
    p = K * g
    res = {}
    for s in (1.0, 2.0, 3.0, 4.0):
        for n in (60, 120, 240, 1000):
            acc, maj, hit_c, hit_l, rho = [], [], [], [], []
            for r in range(reps):
                rng = np.random.default_rng([9191, n, int(s * 10), r])
                A = rng.choice(K, 1)[0]
                idx = np.arange(A * g, (A + 1) * g)

                def draw(m):
                    X = rng.normal(size=(m, p))
                    y = (X[:, idx].sum(1) + s * rng.normal(size=m) > 0).astype(int)
                    return X, y
                Xtr, ytr = draw(n)
                Xva, yva = draw(2000)
                mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=r)).fit(Xtr, ytr)
                acc.append(float(np.mean(mlp.predict(Xva) == yva))); maj.append(float(max(yva.mean(), 1 - yva.mean())))
                c = np.abs([np.corrcoef(Xtr[:, j], ytr)[0, 1] for j in range(p)])
                hit_c.append(len(set(np.argsort(-c)[:g]) & set(idx)) / g)
                lr = LogisticRegression(penalty="l1", solver="liblinear", C=0.3).fit(StandardScaler().fit_transform(Xtr), ytr)
                hit_l.append(len(set(np.argsort(-np.abs(lr.coef_[0]))[:g]) & set(idx)) / g)
                rho.append(float(np.mean(c[idx])))
            res[f"s{s}_n{n}"] = {"mlp_acc_median": float(np.median(acc)), "mlp_acc_p10": float(np.quantile(acc, .1)), "majority_median": float(np.median(maj)),
                                 "screen_hit_rate_corr": float(np.mean(hit_c)), "screen_hit_rate_L1": float(np.mean(hit_l)), "mean_abs_corr_true_indicators": float(np.mean(rho))}
            print(f"s={s} n={n}: MLP {np.median(acc):.2f} (maj {np.median(maj):.2f}) | hits top-{g}: corr {np.mean(hit_c):.2f}, L1 {np.mean(hit_l):.2f} | |corr| verd. {np.mean(rho):.2f}", flush=True)
    json.dump({"reps": reps, "K": K, "g": g, "labels_used": "só treino/validação do gerador em sementes de calibração; sem ontologia, sem TREPAN, sem teste da avaliação", "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
