"""B3-Q / Doc. 48 §3: quanto da partição de conceitos do gerador LATENT_CONCEPT_V1 (Doc. 44) é recuperável SÓ dos dados de treino (agrupamento por |Spearman|, como o braço I)?
Mede a informação da partição que é REDUNDANTE com os dados (ARI vs partição verdadeira) em função de n. Gerador só dos dados X (sem rótulos nem regra). Sem modelos, sem ontologias, sem avaliação."""
import json
import sys
import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.metrics import adjusted_rand_score


def gen(n, K, g, rng, lo=0.60, hi=0.85):
    z = rng.normal(size=(n, K))
    lam = rng.uniform(lo, hi, size=(K, g))
    X = np.empty((n, K * g))
    for c in range(K):
        for j in range(g):
            X[:, c * g + j] = lam[c, j] * z[:, c] + np.sqrt(1 - lam[c, j] ** 2) * rng.normal(size=n)
    return X, np.repeat(np.arange(K), g)


def recover(X, K):
    R = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])])
    d = np.clip(1.0 - np.abs(np.nan_to_num(np.corrcoef(R, rowvar=False))), 0, None)
    np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(d, checks=False), method="average"), t=K, criterion="maxclust")


def main(out):
    res = {}
    for (lo, hi) in ((0.60, 0.85), (0.45, 0.65), (0.35, 0.55), (0.25, 0.45)):
        for (K, g) in ((8, 6), (16, 6)):
            for n in (30, 60, 120, 240, 1000):
                rng = np.random.default_rng([3434, K, n, int(lo * 100)])
                aris = []
                for _ in range(100):
                    X, true = gen(n, K, g, rng, lo, hi)
                    aris.append(adjusted_rand_score(true, recover(X, K)))
                res[f"lam{lo}-{hi}_K{K}_g{g}_n{n}"] = {"ARI_mean": float(np.mean(aris)), "ARI_p10": float(np.quantile(aris, 0.1)), "ARI_p90": float(np.quantile(aris, 0.9))}
    json.dump({"method": "average-linkage on 1-|Spearman|, K clusters (K conhecido); 100 réplicas por célula; só X (sem rótulos, regra ou ontologia); exploração de DESENHO, não de resultados",
               "results": res}, open(out, "w"), indent=1)
    for k, v in res.items():
        if k.endswith("g6_n60") or k.endswith("g6_n120") or k.endswith("g6_n240") or k.endswith("g6_n1000"):
            print(k, round(v["ARI_mean"], 2))


if __name__ == "__main__":
    main(sys.argv[1])
