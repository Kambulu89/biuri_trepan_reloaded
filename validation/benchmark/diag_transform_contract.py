"""Diagnóstico P0 (Doc. 21): contrato do transformador ontológico e mapa de espaços. Só leitura; usa apenas dados de treino.
    python -m validation.benchmark.diag_transform_contract --dataset iris --seed 42 --out FILE.json
Verifica: (1) determinismo e independência de linha/lote/ordem; (2) originais inalteradas; (3) derivadas RECALCULADAS pelo projetor (ignora valores de entrada das
derivadas); (4) comportamento fora do domínio observado (extremos) e com NaN; (5) a transformação das consultas é a MESMA do treino; (6) espaços e dimensões
(original / enriquecido / entrada do oráculo); (7) coerência do controlo permutado com a sua própria definição.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # fixa o ambiente numérico
import numpy as np

from core.benchmark.semantic import OwlSemanticProvider, ShuffledSemanticProvider
from validation.benchmark import datasets as ds_mod


def check_ctx(ctx, Xtr, rng) -> dict:
    n_o, onto = len(ctx.orig_idx), list(ctx.onto_idx)
    Z = ctx.enrich(Xtr)
    out = {"n_original": n_o, "n_derived": len(onto), "enriched_dim": ctx.n_features}
    out["deterministic"] = bool(np.array_equal(Z, ctx.enrich(Xtr)))
    out["originals_unchanged"] = bool(np.array_equal(Z[:, :n_o], np.asarray(Xtr, float)))
    idx = rng.choice(len(Xtr), size=min(60, len(Xtr)), replace=False)
    batch = ctx.enrich(Xtr[idx])
    singles = np.vstack([ctx.enrich(Xtr[i:i + 1]) for i in idx])
    perm = rng.permutation(len(idx))
    out["row_independent_batch_vs_single_max_abs_diff"] = float(np.max(np.abs(batch - singles)))
    out["permutation_invariant_max_abs_diff"] = float(np.max(np.abs(ctx.enrich(Xtr[idx][perm]) - batch[perm])))
    # (3) o projetor recalcula as derivadas a partir das originais (ignora o que vem nas colunas derivadas)
    junk = Z.copy()
    junk[:, onto] = rng.normal(size=(len(Z), len(onto))) * 100.0
    out["projector_recomputes_derived_max_abs_diff"] = float(np.max(np.abs(ctx.query_projector()(junk) - Z))) if onto else 0.0
    # (5) mesma transformação nas consultas e no treino
    out["query_projector_equals_train_transform_max_abs_diff"] = float(np.max(np.abs(ctx.query_projector()(Z) - Z)))
    # (4) fora do domínio: extremos e NaN
    sd = np.maximum(np.asarray(Xtr, float).std(axis=0), 1e-12)
    mu = np.asarray(Xtr, float).mean(axis=0)
    ext = {}
    for k in (3, 10, 50):
        E = mu + k * sd * rng.choice([-1.0, 1.0], size=(30, n_o))
        Ze = ctx.enrich(E)
        ext[f"k={k}sd_finite"] = bool(np.isfinite(Ze).all())
        ext[f"k={k}sd_max_abs_derived_sd_units"] = float(np.max(np.abs(Ze[:, onto] - Z[:, onto].mean(0)) / np.maximum(Z[:, onto].std(0), 1e-12))) if onto else 0.0
    out["extreme_inputs"] = ext
    bad = np.asarray(Xtr[:3], float).copy()
    bad[0, 0] = np.nan
    try:
        ctx.enrich(bad)
        out["nan_input"] = "accepted (propagates)"
    except Exception as exc:      # contrato: valores ausentes exigem imputação explícita
        out["nan_input"] = f"rejects: {type(exc).__name__}"
    return out


def derived_plausibility(ctx, Xtr, Xheld, rng) -> dict:
    """Coerência algébrica ≠ plausibilidade: as derivadas das consultas coerentes (originais amostradas MARGINALMENTE) caem onde, relativamente às derivadas
    das linhas reais? Base de referência: linhas reais de teste-interno (partição de treino não usada no ajuste do transformador)."""
    from core.trepan_original import FeatureDistributionModel
    n_o, onto = len(ctx.orig_idx), list(ctx.onto_idx)
    if not onto:
        return {}
    Ztr = ctx.enrich(Xtr)[:, onto]
    lo, hi, sd = Ztr.min(0), Ztr.max(0), np.maximum(Ztr.std(0), 1e-12)
    def stats(Zd):
        below, above = np.maximum(lo - Zd, 0.0), np.maximum(Zd - hi, 0.0)
        return {"any_derived_out_of_train_range": float(((below > 0) | (above > 0)).any(axis=1).mean()),
                "mean_excess_sd_units": float(((below + above) / sd).mean()), "p99_abs_z": float(np.quantile(np.abs((Zd - Ztr.mean(0)) / sd), 0.99))}
    model = FeatureDistributionModel(random_state=0).fit(np.asarray(Xtr, float))
    synth = ctx.enrich(model.draw(4000))[:, onto]
    return {"real_heldout": stats(ctx.enrich(Xheld)[:, onto]), "synthetic_coherent_root_marginal": stats(synth)}


def run(dataset: str, seed: int) -> dict:
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    rng = np.random.default_rng(seed)
    n = len(ds.X)
    perm_all = rng.permutation(n)
    tr, held = perm_all[: int(0.75 * n)], perm_all[int(0.75 * n):]
    Xtr, Xheld = ds.X[tr], ds.X[held]
    prov = OwlSemanticProvider(str(spec.ontology_path))
    ctx = prov.build(Xtr, ds.feature_names, seed)
    res = {"dataset": dataset, "seed": seed, "available": bool(ctx.available), "ontology_quality": ctx.info.get("ontology_quality"), "reason": ctx.reason}
    if not ctx.available:
        return res
    res["real"] = check_ctx(ctx, Xtr, rng)
    shuf = ShuffledSemanticProvider(prov).from_context(ctx, seed)
    res["shuffled"] = check_ctx(shuf, Xtr, rng)
    res["plausibility_real"] = derived_plausibility(ctx, Xtr, Xheld, rng)
    res["plausibility_shuffled"] = derived_plausibility(shuf, Xtr, Xheld, rng)
    res["spaces"] = {"dataset_original_dim": int(Xtr.shape[1]), "tree_space_enriched_dim": ctx.n_features, "oracle_input_dim": int(Xtr.shape[1]),
                     "oracle_sees_columns": "orig_idx = 0..p-1 (OriginalOracleProjection)", "queries_reach_oracle_as": "enriched rows projected to original columns",
                     "preprocessing": "none (numeric matrix as loaded); scaling lives inside the MLP pipeline fitted on train"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    Path(a.out).write_text(json.dumps(run(a.dataset, a.seed), indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
