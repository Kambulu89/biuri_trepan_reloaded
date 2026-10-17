"""B3-Q / Doc. 61: DEPENDÊNCIA DO MÉTODO GENÉRICO no limiar τ_s. Mede a taxa de identificação dos g indicadores verdadeiros (família aditiva R8) com estimadores supervisionados genéricos diferentes e
com o melhor-de-todos, em sementes NOVAS (base 9800) e células s∈{2,3}, n∈{60,120,240}. Só gerador + estimadores genéricos; sem ontologia, TREPAN, braços OWL ou teste da avaliação."""
import json
import sys
import warnings

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler


def main(out, reps=20, K=8, g=6):
    warnings.filterwarnings("ignore")
    p = K * g
    res = {}
    for s in (2.0, 3.0):
        for n in (60, 120, 240):
            hits = {k: [] for k in ("corr", "L1_C0.3", "L1_CV", "mutual_info", "random_forest", "best_of_all")}
            for r in range(reps):
                rng = np.random.default_rng([9800, n, int(s * 10), r])
                A = rng.choice(K, 1)[0]
                idx = np.arange(A * g, (A + 1) * g)
                X = rng.normal(size=(n, p))
                y = (X[:, idx].sum(1) + s * rng.normal(size=n) > 0).astype(int)
                Z = StandardScaler().fit_transform(X)
                score = {
                    "corr": np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(p)]),
                    "L1_C0.3": np.abs(LogisticRegression(penalty="l1", solver="liblinear", C=0.3).fit(Z, y).coef_[0]),
                    "L1_CV": np.abs(LogisticRegressionCV(penalty="l1", solver="liblinear", Cs=8, cv=3, random_state=r).fit(Z, y).coef_[0]),
                    "mutual_info": mutual_info_classif(X, y, random_state=r),
                    "random_forest": RandomForestClassifier(n_estimators=150, random_state=r).fit(X, y).feature_importances_}
                h = {k: len(set(np.argsort(-v)[:g]) & set(idx)) / g for k, v in score.items()}
                for k, v in h.items():
                    hits[k].append(v)
                hits["best_of_all"].append(max(h.values()))
            res[f"s{s}_n{n}"] = {k: {"mean": float(np.mean(v)), "se": float(np.std(v, ddof=1) / np.sqrt(len(v)))} for k, v in hits.items()}
            print(f"s={s} n={n}: " + ", ".join(f"{k} {np.mean(v):.2f}" for k, v in hits.items()), flush=True)
    json.dump({"reps": reps, "seed_base": 9800, "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
