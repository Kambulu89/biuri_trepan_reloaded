"""Auditoria da representação enriquecida (OWL real vs permutada vs convencional) — Doc. 25. EXPERIMENTAL; só leitura; só treino + rótulos do oráculo.

Distingue conhecimento ontológico VÁLIDO (matching, grupos alinhados com a estrutura dos dados) de features NUMERICAMENTE ÚTEIS. Para uma derivada D = f(X) determinística,
a informação mútua condicional I(Y; D | X) = 0 por construção (nada novo para o MLP, que só vê X): a utilidade possível é para a ÁRVORE (limiares axiais sobre
combinações de colunas) — medida aqui pelo ganho de informação do melhor limiar axial em D vs nas originais.
    python -m validation.benchmark.audit_representation --dataset iris --seed 42 --out FILE.json
"""
from __future__ import annotations

from validation.benchmark import run as _R  # noqa: F401

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def _best_axis_ig(Z, y, cols, n_thr=24):
    from core.trepan_original import _information_gain
    classes = np.unique(y)
    best = 0.0
    for j in cols:
        v = Z[:, j]
        qs = np.unique(np.quantile(v, np.linspace(0.05, 0.95, n_thr)))
        for t in qs:
            ig = _information_gain(y, v <= t, classes)
            best = max(best, float(ig))
    return best


def sources_of(ctx, Xtr, rng):
    """Fontes de cada derivada por sensibilidade (caixa-preta, genérico): perturbar cada coluna original e ver se a derivada muda."""
    n_o = len(ctx.orig_idx)
    base = ctx.enrich(Xtr[:20])
    sd = np.maximum(Xtr.std(axis=0), 1e-12)
    src = {k: [] for k in ctx.onto_idx}
    for j in range(n_o):
        pert = Xtr[:20].copy()
        pert[:, j] += sd[j]
        diff = np.abs(ctx.enrich(pert) - base).max(axis=0)
        for k in ctx.onto_idx:
            if diff[k] > 1e-9:
                src[k].append(j)
    return src


def group_alignment(Xtr, groups, rng, n_null=200):
    """|Spearman| médio dentro de cada grupo de fontes vs nulo (grupos aleatórios do mesmo tamanho): z-score da média."""
    from scipy.stats import rankdata
    R = np.column_stack([rankdata(Xtr[:, j]) for j in range(Xtr.shape[1])])
    C = np.abs(np.nan_to_num(np.corrcoef(R, rowvar=False)))
    def mean_pair(g):
        if len(g) < 2:
            return np.nan
        sub = C[np.ix_(g, g)]
        return float((sub.sum() - len(g)) / (len(g) * (len(g) - 1)))
    obs, z = [], []
    for g in groups:
        if len(g) < 2:
            continue
        o = mean_pair(g)
        null = np.array([mean_pair(list(rng.choice(Xtr.shape[1], len(g), replace=False))) for _ in range(n_null)])
        obs.append(o)
        z.append((o - null.mean()) / max(null.std(), 1e-9))
    return {"mean_abs_spearman_within_groups": float(np.nanmean(obs)) if obs else None, "mean_z_vs_random_groups": float(np.mean(z)) if z else None,
            "share_groups_above_null_p95": float(np.mean(np.array(z) > 1.645)) if z else None, "n_groups_ge2": len(obs)}


def audit_ctx(name, ctx, Xtr, y, rng) -> dict:
    from sklearn.feature_selection import mutual_info_classif
    from scipy.stats import rankdata
    n_o, onto = len(ctx.orig_idx), list(ctx.onto_idx)
    Z = ctx.enrich(Xtr)
    out = {"name": name, "n_derived": len(onto)}
    if not onto:
        return out
    src = sources_of(ctx, Xtr, rng)
    groups = [src[k] for k in onto]
    sizes = [len(g) for g in groups]
    out["source_group_sizes"] = {"min": int(min(sizes)), "median": float(np.median(sizes)), "max": int(max(sizes))}
    out["groups_alignment_with_data_structure"] = group_alignment(Xtr, groups, rng)
    # redundância
    max_corr_single = []
    for k in onto:
        col = Z[:, k]
        c = [abs(np.corrcoef(col, Z[:, j])[0, 1]) for j in range(n_o) if np.std(Z[:, j]) > 1e-12 and np.std(col) > 1e-12]
        max_corr_single.append(max(c) if c else 0.0)
    out["share_derived_nearly_a_single_original(|r|>0.95)"] = float(np.mean(np.array(max_corr_single) > 0.95))
    # informação
    mi_o = mutual_info_classif(Z[:, :n_o], y, random_state=0)
    mi_d = mutual_info_classif(Z[:, onto], y, random_state=0)
    best_member = np.array([mi_o[g].max() if len(g) else 0.0 for g in groups])
    out["mi_derived_median"] = float(np.median(mi_d)); out["mi_original_median"] = float(np.median(mi_o))
    out["share_derived_mi_above_best_member"] = float(np.mean(mi_d > best_member))
    out["conditional_info_Y_D_given_X"] = 0.0           # D é função determinística de X (por construção)
    # utilidade para a árvore: melhor limiar axial em derivadas vs originais (raiz e sub-amostras bootstrap)
    ig_o = _best_axis_ig(Z, y, range(n_o)); ig_d = _best_axis_ig(Z, y, onto)
    wins, gain = 0, []
    for b in range(15):
        idx = rng.choice(len(Z), size=max(60, len(Z) // 2), replace=True)
        go, gd = _best_axis_ig(Z[idx], y[idx], range(n_o), 12), _best_axis_ig(Z[idx], y[idx], onto, 12)
        wins += gd > go
        gain.append(gd - go)
    out["tree_utility"] = {"root_best_ig_original": ig_o, "root_best_ig_derived": ig_d, "share_bootstrap_subsamples_where_derived_wins": wins / 15,
                           "mean_ig_gain_derived_minus_original": float(np.mean(gain))}
    # espaço de candidatos
    from core.trepan_original import TrepanOriginalClassifier
    t = TrepanOriginalClassifier(max_features_per_node=12, random_state=0); t.classes_ = np.unique(y)
    out["candidates"] = {"original_space": len(t._candidate_literals(Xtr, y)), "enriched_space": len(t._candidate_literals(Z, y)), "p": n_o, "p_plus_k": Z.shape[1]}
    out["semantic_structure"] = {"weights_nonunit": int(np.count_nonzero(np.asarray(ctx.weights) != 1.0)), "groups_named": int(len({g for g in (ctx.groups or []) if g})),
                                 "relatedness_offdiag_nonzero": int(np.count_nonzero(np.triu(np.asarray(ctx.relatedness) - np.eye(len(ctx.relatedness)), 1))),
                                 "depth_mean_derived": float(np.mean(np.asarray(ctx.depths)[onto]))}
    return out


def run(dataset: str, seed: int) -> dict:
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from core.benchmark.semantic import ConventionalFeatureProvider, OwlSemanticProvider, ShuffledSemanticProvider
    from validation.benchmark import datasets as ds_mod
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    Xtr, _Xte, ytr, _ = train_test_split(ds.X, ds.y, test_size=0.25, stratify=ds.y, random_state=seed)      # teste não usado
    oracle = make_pipeline(StandardScaler(), MLPClassifier((64,), max_iter=600, random_state=seed)).fit(Xtr, ytr)
    y = np.asarray(oracle.predict(Xtr))
    rng = np.random.default_rng(seed)
    prov = OwlSemanticProvider(str(spec.ontology_path))
    real = prov.build(Xtr, ds.feature_names, seed)
    res = {"dataset": dataset, "seed": seed, "matching": {"available": bool(real.available), "reason": real.reason, "ontology_quality": real.info.get("ontology_quality"),
                                                         "mapped_features": real.info.get("mapped_feature_count"), "unmapped_features": real.info.get("unmapped_feature_count"),
                                                         "feature_coverage": real.info.get("feature_coverage"), "reasoner_consistent": real.info.get("reasoner_consistent"),
                                                         "graph_nodes": real.info.get("n_ontology_graph_nodes"), "n_original": int(Xtr.shape[1])}}
    if not real.available:
        return res
    shuf = ShuffledSemanticProvider(prov).from_context(real, seed)
    conv = ConventionalFeatureProvider(len(real.onto_idx)).build(Xtr, ds.feature_names, seed)
    res["real"] = audit_ctx("owl_real", real, Xtr, y, rng)
    res["shuffled"] = audit_ctx("owl_shuffled", shuf, Xtr, y, rng)
    if conv.available:
        res["conventional"] = audit_ctx("conventional", conv, Xtr, y, rng)
    # estabilidade das derivadas entre ajustes do transformador em subamostras de treino (mesmos dados de referência)
    ref = ds.X
    a = real.enrich(ref)[:, real.onto_idx]
    sub = np.random.default_rng(seed + 1).choice(len(Xtr), size=int(0.8 * len(Xtr)), replace=False)
    b_ctx = prov.build(Xtr[sub], ds.feature_names, seed)
    if b_ctx.available:
        b = b_ctx.enrich(ref)[:, b_ctx.onto_idx]
        res["stability_under_80pct_refit"] = {"min_corr": float(min(abs(np.corrcoef(a[:, i], b[:, i])[0, 1]) for i in range(a.shape[1]))),
                                              "max_rel_std_change": float(np.max(np.abs(b.std(0) - a.std(0)) / np.maximum(a.std(0), 1e-12)))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(run(a.dataset, a.seed), indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
