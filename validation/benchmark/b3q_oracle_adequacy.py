"""B3-Q / Doc. 57 (gate G1b): «acima da maioritária» NÃO basta para dizer que o oráculo aprendeu a estrutura certa. Mede a ADEQUAÇÃO ESTRUTURAL do MLP na família aditiva (R8): fração da importância
por permutação (accuracy, validação gerada de novo, 2000 exemplos) que cai nos g indicadores verdadeiros, vs a fração de acaso g/p. Só gerador + MLP; sementes de calibração (base 9700); sem ontologia,
TREPAN, braços OWL nem teste da avaliação."""
import json
import sys
import warnings

import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def main(out, reps=15, K=8, g=6):
    warnings.filterwarnings("ignore")
    p = K * g
    res = {}
    for s in (1.0, 2.0, 3.0):
        for n in (60, 120, 240, 1000):
            share = []
            for r in range(reps):
                rng = np.random.default_rng([9700, n, int(s * 10), r])
                A = rng.choice(K, 1)[0]
                idx = np.arange(A * g, (A + 1) * g)
                def draw(m):
                    X = rng.normal(size=(m, p)); return X, (X[:, idx].sum(1) + s * rng.normal(size=m) > 0).astype(int)
                Xtr, ytr = draw(n); Xva, yva = draw(2000)
                mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=800, random_state=r)).fit(Xtr, ytr)
                imp = permutation_importance(mlp, Xva, yva, n_repeats=3, random_state=r).importances_mean
                imp = np.clip(imp, 0, None)
                share.append(float(imp[idx].sum() / imp.sum()) if imp.sum() > 0 else float("nan"))
            sh = np.array(share)
            res[f"s{s}_n{n}"] = {"importance_share_true_indicators_median": float(np.nanmedian(sh)), "p10": float(np.nanquantile(sh, .1)), "p90": float(np.nanquantile(sh, .9)), "chance_share": g / p}
            print(f"s={s} n={n}: share verd. {np.nanmedian(sh):.2f} (p10 {np.nanquantile(sh,.1):.2f}) vs acaso {g/p:.3f}", flush=True)
    json.dump({"reps": reps, "seed_base": 9700, "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
