"""Estabilidade da seleção de divisões entre réplicas de amostragem (Doc. 22). EXPERIMENTAL; só leitura; só DEV; orçamento fixo (sem tuning estrutural).

Para cada braço (histórico/coerente × consultas ativas ligadas/desligadas) ajusta ``n_rep`` árvores na MESMA unidade (mesmo split, mesmo FrozenOracle) variando só
``random_state`` (sementes de amostragem) e mede: Jaccard médio par-a-par dos conjuntos de features usadas nas primeiras ``k`` divisões e em toda a árvore, e o DP da
fidelidade ao oráculo no treino (sem teste).
    python -m validation.benchmark.selection_stability --unit iris 42 --out FILE.json
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # fixa o ambiente numérico
import numpy as np

from core.benchmark import runner as RN

CAP: dict = {}


class _Stop(Exception):
    pass


def jaccard(a, b):
    a, b = set(a), set(b)
    return 1.0 if not a and not b else len(a & b) / len(a | b)


def feats(tree, k=None):
    out = []
    for r in tree.iter_nodes():
        pass
    seq = [n for n in tree.iter_nodes() if getattr(n, "test", None) is not None]
    seq.sort(key=lambda n: n.node_id)
    seq = seq[:k] if k else seq
    return {int(l.feature) for n in seq for l in n.test.literals}


def run(dataset: str, seed: int, n_rep: int = 3, k: int = 5) -> dict:
    from validation.benchmark import datasets as ds_mod
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    cfg = dataclasses.replace(R.benchmark_config(seed), structure_tuning=False, coherent_pilot=True)
    orig = RN.BenchmarkRunner._run_arm

    def cap(self, spec_, ds_, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, *a, **kw):
        if spec_.arm_id == "trepan_original_enr" and "ctx" not in CAP:
            CAP.update(ctx=ctx_real, Xtr=Xtr, fo=a[7], budget=a[3])      # a = (oracle_preds, infos, selected, budget, seed, labels, minority, fo, fe)
            raise _Stop()
        return orig(self, spec_, ds_, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, *a, **kw)
    RN.BenchmarkRunner._run_arm = cap
    arms = [x for x in RN.default_arms(cfg) if x.arm_id in ("mlp_original", "trepan_original_enr")]
    try:
        try:
            RN.BenchmarkRunner(cfg, arms).run(ds, str(spec.ontology_path) if spec.ontology_path else None)
        except _Stop:
            pass
    finally:
        RN.BenchmarkRunner._run_arm = orig
    ctx, Xtr, fo, budget = CAP["ctx"], CAP["Xtr"], CAP["fo"], CAP["budget"]
    from core.controlled_trepan_experiment import OriginalOracleProjection
    oracle = OriginalOracleProjection(fo, ctx.orig_idx)
    Z = ctx.enrich(Xtr)
    ytrain = np.asarray(oracle.predict(Z))
    res = {"dataset": dataset, "seed": seed, "n_rep": n_rep, "k_first_splits": k, "structure_tuning": False, "arms": {}}
    from core.trepan_reloaded_historical import TrepanReloadedClassifier as T
    arm_cfg = {"hist_active_on": dict(coherent=False, active=True), "hist_active_off": dict(coherent=False, active=False),
               "coh_active_on": dict(coherent=True, active=True), "coh_active_off": dict(coherent=True, active=False)}
    for name, c in arm_cfg.items():
        trees = []
        for r in range(n_rep):
            kw = dict(budget, random_state=seed + 1000 * r, alpha=0.35, beta=0.2, mirror_when_no_semantic_effect=False,
                      semantic_active_query_fraction=0.65 if c["active"] else 0.0, coherent_sampling=c["coherent"])
            fit_kw = dict(semantic_feature_weights=np.asarray(ctx.weights), semantic_feature_groups=list(ctx.groups),
                          semantic_relatedness_matrix=np.asarray(ctx.relatedness), semantic_feature_depths=np.asarray(ctx.depths), query_projector=ctx.query_projector())
            if c["coherent"]:
                fit_kw["coherent_original_indices"] = list(ctx.orig_idx)
            t = T(**kw).fit(Z, oracle=oracle, feature_names=ctx.enriched_names, **fit_kw)
            trees.append((t, float(np.mean(np.asarray(t.predict(Z)) == ytrain))))
        pairs = list(itertools.combinations(range(n_rep), 2))
        res["arms"][name] = {"jaccard_first_k": float(np.mean([jaccard(feats(trees[i][0], k), feats(trees[j][0], k)) for i, j in pairs])),
                             "jaccard_all": float(np.mean([jaccard(feats(trees[i][0]), feats(trees[j][0])) for i, j in pairs])),
                             "train_fidelity_mean": float(np.mean([f for _, f in trees])), "train_fidelity_sd": float(np.std([f for _, f in trees])),
                             "nodes_mean": float(np.mean([t.node_count_ for t, _ in trees]))}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(run(a.unit[0], int(a.unit[1])), indent=1), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
