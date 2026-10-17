"""B3-Q / Doc. 49 §3: protocolo da PORTA DE RECUPERABILIDADE da partição de conceitos (só X de treino; nunca rótulos nem teste; nenhuma ontologia, nenhum modelo TREPAN).
Compara métodos não supervisionados plausíveis (agrupamento hierárquico por |Spearman| e por |Pearson|, espectral sobre |Pearson|, k-means sobre o perfil de correlação, análise fatorial com
varimax) com K conhecido e um método com K estimado (silhueta). Mede: ARI global; pureza média por conceito (recuperação PARCIAL dos conceitos); variabilidade entre réplicas (p10/p50/p90);
sensibilidade ao método (melhor-de-todos = o mais favorável aos DADOS, logo o mais conservador para o conhecimento externo). Gerador = LATENT_CONCEPT_V1 (Doc. 44) só nos indicadores X."""
from __future__ import annotations

import json
import sys
import warnings
from typing import Callable, Dict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import adjusted_rand_score, silhouette_score

from validation.benchmark.b3q_partition_recoverability import gen


def _corr(X, spearman):
    Z = np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])]) if spearman else X
    C = np.nan_to_num(np.corrcoef(Z, rowvar=False))
    return np.abs(C)


def m_hier_spearman(X, K):
    d = 1.0 - _corr(X, True); np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(np.clip(d, 0, None), checks=False), "average"), K, "maxclust")


def m_hier_pearson(X, K):
    d = 1.0 - _corr(X, False); np.fill_diagonal(d, 0.0)
    return fcluster(linkage(squareform(np.clip(d, 0, None), checks=False), "average"), K, "maxclust")


def m_spectral(X, K):
    A = _corr(X, False); np.fill_diagonal(A, 0.0)
    return SpectralClustering(K, affinity="precomputed", random_state=0, assign_labels="kmeans").fit_predict(A + 1e-9)


def m_kmeans_profile(X, K):
    C = _corr(X, False)
    return KMeans(K, n_init=10, random_state=0).fit_predict(C)


def m_factor_varimax(X, K):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    fa = FactorAnalysis(n_components=K, rotation="varimax", random_state=0).fit(Z)
    return np.argmax(np.abs(fa.components_), axis=0)


def m_hier_silhouette(X, K):
    d = 1.0 - _corr(X, True); np.fill_diagonal(d, 0.0)
    Lk = linkage(squareform(np.clip(d, 0, None), checks=False), "average")
    best, lab = -2.0, None
    for k in range(max(2, K // 2), min(X.shape[1] - 1, 2 * K) + 1):
        l = fcluster(Lk, k, "maxclust")
        if len(set(l)) < 2:
            continue
        s = silhouette_score(d, l, metric="precomputed")
        if s > best:
            best, lab = s, l
    return lab


METHODS: Dict[str, Callable] = {"hier_spearman": m_hier_spearman, "hier_pearson": m_hier_pearson, "spectral_abs_pearson": m_spectral, "kmeans_corr_profile": m_kmeans_profile,
                                "factor_varimax": m_factor_varimax, "hier_spearman_K_estimated": m_hier_silhouette}


def concept_purity(true, lab):
    """Recuperação PARCIAL: para cada conceito, fração dos seus indicadores no cluster que contém mais deles; média sobre conceitos."""
    out = []
    for c in np.unique(true):
        m = lab[true == c]
        out.append(np.bincount(m).max() / len(m))
    return float(np.mean(out))


def run(out, reps=40, K=8, g=6):
    warnings.filterwarnings("ignore")
    res = {}
    for (lo, hi) in ((0.60, 0.85), (0.45, 0.65), (0.35, 0.55), (0.25, 0.45)):
        for n in (30, 60, 120, 240, 1000):
            per = {m: {"ari": [], "purity": []} for m in METHODS}
            for r in range(reps):
                rng = np.random.default_rng([6161, K, n, int(lo * 100), r])
                X, true = gen(n, K, g, rng, lo, hi)
                for m, f in METHODS.items():
                    lab = f(X, K)
                    per[m]["ari"].append(adjusted_rand_score(true, lab)); per[m]["purity"].append(concept_purity(true, lab))
            cell = {m: {"ari_p10": float(np.quantile(v["ari"], .1)), "ari_median": float(np.median(v["ari"])), "ari_p90": float(np.quantile(v["ari"], .9)),
                        "purity_median": float(np.median(v["purity"]))} for m, v in per.items()}
            best_med = max(c["ari_median"] for c in cell.values())
            cell["_best_of_methods_median_ari"] = best_med
            cell["_best_of_methods_median_purity"] = max(c["purity_median"] for k, c in cell.items() if not k.startswith("_"))
            cell["_range_of_median_ari_across_methods"] = [min(c["ari_median"] for k, c in cell.items() if not k.startswith("_")), best_med]
            res[f"lam{lo}-{hi}_K{K}_g{g}_n{n}"] = cell
            print(f"lam{lo}-{hi} n={n}: best-of-methods median ARI {best_med:.2f}; per method " + ", ".join(f"{m[:8]}={c['ari_median']:.2f}" for m, c in cell.items() if not m.startswith('_')), flush=True)
    json.dump({"protocol": __doc__, "reps": reps, "K": K, "g": g, "labels_used": False, "test_data_used": False, "results": res}, open(out, "w"), indent=1)


if __name__ == "__main__":
    run(sys.argv[1], reps=int(sys.argv[2]) if len(sys.argv) > 2 else 40)
