"""Diagnóstico B2.0 (EXPERIMENTAL; só leitura; sem seleção): (A) representação OWL e (B) consultas ativas/projeção, em DEV.

* Usa só treino e rótulos do ORÁCULO congelado (nunca o teste nem os rótulos reais para qualquer decisão).
* Tuning estrutural desligado (orçamento fixo do TreeBudget) porque o diagnóstico NÃO mede desempenho final; documentado em ``meta``.
* Agnóstico ao dataset: nenhuma regra depende da identidade do dataset.
    python -m validation.benchmark.diag_repr_queries --unit iris 42 --out FILE.json
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico
import numpy as np

from core.benchmark import runner as RN
from validation.benchmark import datasets as ds_mod
from validation.benchmark import manifest as mf

CAP: dict = {}


def _safe_corr(a, b):
    sa, sb = np.std(a), np.std(b)
    if sa < 1e-12 or sb < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def representation_report(ctx, Xtr, oracle_pred, seed: int) -> dict:
    from sklearn.feature_selection import mutual_info_classif
    from sklearn.linear_model import LinearRegression
    n_o, onto = len(ctx.orig_idx), list(ctx.onto_idx)
    Z = ctx.enrich(Xtr)
    out = {"n_original": int(n_o), "n_derived": len(onto), "derived_names": [ctx.enriched_names[i] for i in onto][:40],
           "finite_all": bool(np.isfinite(Z).all()), "originals_unchanged": bool(np.array_equal(Z[:, :n_o], np.asarray(Xtr, float))),
           "deterministic": bool(np.array_equal(Z, ctx.enrich(Xtr)))}
    proj = ctx.query_projector()
    out["projector_fixpoint_max_abs_err"] = float(np.max(np.abs(proj(Z) - Z))) if len(Z) else 0.0
    # redundância / determinação por originais
    O = Z[:, :n_o]
    der = []
    mi_orig = mutual_info_classif(O, oracle_pred, random_state=seed)
    rng = np.random.default_rng(seed)
    null = np.concatenate([mutual_info_classif(O, rng.permutation(oracle_pred), random_state=seed + k) for k in range(3)])
    thr = float(np.quantile(null, 0.95))
    mi_der = mutual_info_classif(Z[:, onto], oracle_pred, random_state=seed) if onto else np.array([])
    for k, j in enumerate(onto):
        col = Z[:, j]
        corrs = [abs(_safe_corr(col, O[:, i])) for i in range(n_o)]
        r2 = float(LinearRegression().fit(O, col).score(O, col)) if np.std(col) > 1e-12 else 1.0
        der.append({"name": ctx.enriched_names[j], "std": float(np.std(col)), "mean": float(np.mean(col)), "min": float(col.min()), "max": float(col.max()),
                    "max_abs_corr_with_original": float(max(corrs) if corrs else 0.0), "r2_from_originals": r2,
                    "near_constant": bool(np.std(col) < 1e-9), "mutual_info_with_oracle": float(mi_der[k]), "mi_above_null95": bool(mi_der[k] > thr)})
    out["derived"] = der
    out["originals_scale"] = {"std_median": float(np.median(np.std(O, axis=0))), "mi_median": float(np.median(mi_orig)), "mi_null95": thr}
    out["derived_summary"] = {
        "share_near_duplicate_r2_gt_0.95": float(np.mean([d["r2_from_originals"] > 0.95 for d in der])) if der else None,
        "share_mi_above_null95": float(np.mean([d["mi_above_null95"] for d in der])) if der else None,
        "share_near_constant": float(np.mean([d["near_constant"] for d in der])) if der else None,
        "mi_median": float(np.median([d["mutual_info_with_oracle"] for d in der])) if der else None,
        "std_median": float(np.median([d["std"] for d in der])) if der else None}
    return out


def consistency_of_draws(ctx, model_enr, n: int, projector) -> dict:
    """Features derivadas num sorteio marginal (Original na representação enriquecida) vs recalculadas a partir das colunas originais."""
    n_o, onto = len(ctx.orig_idx), list(ctx.onto_idx)
    if not onto:
        return {}
    raw = model_enr._draw_unconstrained(n)
    recomputed = ctx.enrich(raw[:, :n_o])
    scale = np.maximum(np.std(recomputed[:, onto], axis=0), 1e-9)
    err = np.abs(raw[:, onto] - recomputed[:, onto]) / scale
    proj = projector(raw)
    err_p = np.abs(proj[:, onto] - recomputed[:, onto]) / scale
    return {"marginal_draw_derived_inconsistency_mean_std_units": float(np.mean(err)),
            "marginal_draw_derived_inconsistency_p90": float(np.quantile(err, 0.9)),
            "projected_draw_derived_inconsistency_mean": float(np.mean(err_p))}


def run_diag(dataset: str, seed: int) -> dict:
    spec = ds_mod.REGISTRY[dataset]
    ds, _t, _c = spec.load()
    orig_cfg = R.benchmark_config
    cfg = dataclasses.replace(orig_cfg(seed), structure_tuning=False, b2c_pilot=True)
    orig_run_arm = RN.BenchmarkRunner._run_arm

    def cap_run_arm(self, spec_, ds_, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected,
                    budget, seed_, labels, minority, fo=None, fe=None, oracle_train_pred=None, structural=None):
        if spec_.arm_id == "trepan_original_enr" and "ctx" not in CAP:
            CAP.update(ctx=ctx_real, shuf=ctx_shuf, Xtr=Xtr, fo=fo, budget=budget, seed=seed_, oracle_train_pred=oracle_train_pred)
            raise _Stop()
        return orig_run_arm(self, spec_, ds_, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, oracle_preds, infos, selected,
                            budget, seed_, labels, minority, fo=fo, fe=fe, oracle_train_pred=oracle_train_pred, structural=structural)

    class _Stop(Exception):
        pass

    RN.BenchmarkRunner._run_arm = cap_run_arm
    arms = [a for a in RN.default_arms(cfg) if a.arm_id in ("mlp_original", "trepan_original_enr")]
    try:
        try:
            RN.BenchmarkRunner(cfg, arms).run(ds, str(spec.ontology_path) if spec.ontology_path else None)
        except _Stop:
            pass
    finally:
        RN.BenchmarkRunner._run_arm = orig_run_arm
    ctx, Xtr, fo, budget, sd = CAP["ctx"], CAP["Xtr"], CAP["fo"], CAP["budget"], CAP["seed"]
    from core.benchmark.runner import OriginalOracleProjection
    oracle_enr = OriginalOracleProjection(fo, ctx.orig_idx)
    Z = ctx.enrich(Xtr)
    yz = np.asarray(oracle_enr.predict(Z))
    yo = np.asarray(fo.predict(Xtr))
    res = {"dataset": dataset, "seed": seed, "meta": {"structure_tuning": False, "budget": {k: budget[k] for k in ("min_sample", "max_queries", "max_nodes", "max_depth")},
                                                   "uses_test": False, "uses_real_labels": False},
           "available": bool(ctx.available)}
    if not ctx.available:
        res["reason"] = ctx.reason
        return res
    res["oracle_projection_matches"] = bool(np.array_equal(yz, yo))
    res["representation"] = representation_report(ctx, Xtr, yo, sd)
    # --- candidatos e prioridade no nó raiz: Original em espaço original vs enriquecido
    from core.trepan_original import TrepanOriginalClassifier, FeatureDistributionModel
    common = dict(budget, random_state=sd)
    t_o = TrepanOriginalClassifier(**common)
    t_o.classes_ = np.unique(yo)
    cand_o = t_o._candidate_literals(np.asarray(Xtr, float), yo)
    t_e = TrepanOriginalClassifier(**common)
    t_e.classes_ = np.unique(yo)
    cand_e = t_e._candidate_literals(Z, yo)
    top_e = [lit.feature for _s, lit in cand_e[:12]]
    n_o = len(ctx.orig_idx)
    res["candidate_space"] = {"n_candidates_original_space": len(cand_o), "n_candidates_enriched_space": len(cand_e),
                              "root_top12_share_derived": float(np.mean([f >= n_o for f in top_e])) if top_e else None,
                              "root_best_is_derived": bool(cand_e[0][1].feature >= n_o) if cand_e else None,
                              "root_best_ig_original_space": float(cand_o[0][0]) if cand_o else None,
                              "root_best_ig_enriched_space": float(cand_e[0][0]) if cand_e else None}
    model_enr = FeatureDistributionModel(random_state=sd).fit(Z)
    res["draw_consistency"] = consistency_of_draws(ctx, model_enr, 4000, ctx.query_projector())
    # --- consultas: ajuste B2-C inerte com captura das amostras por nó
    from core.trepan_reloaded_historical import TrepanReloadedClassifier
    store = []

    class _Cap(TrepanReloadedClassifier):
        def _decision_sample_b2c(self, node):
            Xo, yo_, model = super()._decision_sample_b2c(node)
            alt = None
            nb = int(len(Xo) - len(node.real_y))
            if nb > 0:        # amostra-base ALTERNATIVA (mesmo procedimento do Original, outra semente): nula da variabilidade amostral
                import copy
                m2 = copy.copy(model)
                m2._rng = np.random.default_rng(np.random.SeedSequence([int(self.random_state), 0xD1A6, int(node.node_id)]))
                qa = m2.draw(nb, node.constraints)
                alt = (np.vstack([node.real_X, qa]), np.concatenate([node.real_y, np.asarray(self.oracle_.predict(qa))]))
            store.append((node.node_id, int(len(node.real_y)), Xo, yo_, self._b2c_sem_X, self._b2c_sem_y, node.depth, alt))
            return Xo, yo_, model

    INERT = dict(error_focus_min_local_fidelity_gain=10.0, error_focus_fidelity_tolerance=0.0, mirror_when_no_semantic_effect=False)
    kw = dict(common, alpha=0.35, beta=0.2, statistical_fallback=True, **INERT)
    fit_kw = dict(semantic_feature_weights=np.asarray(ctx.weights), semantic_feature_groups=list(ctx.groups),
                  semantic_relatedness_matrix=np.asarray(ctx.relatedness), semantic_feature_depths=np.asarray(ctx.depths), query_projector=ctx.query_projector())
    tree = _Cap(**kw).fit(Z, oracle=oracle_enr, feature_names=ctx.enriched_names, **fit_kw)
    from scipy.stats import ks_2samp
    from core.error_focused_semantic_refinement import split_surrogate_fidelity
    rows = []
    for nid, nr, Xo, yo_, Xs, ys, depth, alt in store:
        qo, qs = Xo[nr:], Xs[nr:]
        if len(qo) < 8 or len(qs) < 8:
            continue
        ks = float(np.mean([ks_2samp(qo[:, j], qs[:, j]).statistic for j in range(Z.shape[1])]))
        sc = np.maximum(np.std(Z, axis=0), 1e-9)
        ref = Z[:, :]
        # proximidade ao melhor limiar (por feature) da amostra real do nó
        real = Xo[:nr]
        ranked = tree._candidate_literals(real, yo_[:nr]) if nr >= 4 else []
        def near(Q):
            if not ranked:
                return None
            d = np.min([np.abs(Q[:, lit.feature] - lit.threshold) / sc[lit.feature] for _s, lit in ranked[:6]], axis=0)
            return float(np.mean(d < 0.25))
        ent = lambda y: float(-(lambda p: (p * np.log2(p)).sum())(np.bincount(np.searchsorted(tree.classes_, y), minlength=len(tree.classes_))[np.bincount(np.searchsorted(tree.classes_, y), minlength=len(tree.classes_)) > 0] / len(y)))
        row = {"node_id": nid, "depth": depth, "n_real": nr, "n_base_queries": int(len(qo)), "n_sem_queries": int(len(qs)), "mean_ks_base_vs_sem": ks,
               "near_boundary_base": near(qo), "near_boundary_sem": near(qs), "label_entropy_base": ent(yo_[nr:]), "label_entropy_sem": ent(ys[nr:])}
        # viés de amostra: o melhor teste estatístico escolhido em D_o e o escolhido em D_s (pesquisa do Original, sem pesos), cada um avaliado em ambas
        t_e._split_counters_ = {}
        res_o = TrepanOriginalClassifier._best_mofn(t_e, Xo, yo_)
        res_s = TrepanOriginalClassifier._best_mofn(t_e, Xs, ys)
        fid = lambda t, X_, y_: float(split_surrogate_fidelity(y_, t.evaluate(X_), tree.classes_)) if t is not None else None
        to, ts = (res_o[0] if res_o else None), (res_s[0] if res_s else None)
        row.update(same_choice_Do_vs_Ds=bool(tree._test_signature(to) == tree._test_signature(ts)),
                   fid_choiceDo_on_Do=fid(to, Xo, yo_), fid_choiceDo_on_Ds=fid(to, Xs, ys), fid_choiceDs_on_Do=fid(ts, Xo, yo_), fid_choiceDs_on_Ds=fid(ts, Xs, ys))
        if alt is not None:
            res_a = TrepanOriginalClassifier._best_mofn(t_e, alt[0], alt[1])
            ta = res_a[0] if res_a else None
            row.update(same_choice_Do_vs_Dalt=bool(tree._test_signature(to) == tree._test_signature(ta)), fid_choiceDalt_on_real=fid(ta, Xo[:nr], yo_[:nr]),
                       fid_choiceDo_on_Dalt=fid(to, alt[0], alt[1]), fid_choiceDalt_on_Dalt=fid(ta, alt[0], alt[1]))
        if to is not None and ts is not None:
            real_X, real_y = Xo[:nr], yo_[:nr]
            row.update(fid_choiceDo_on_real=fid(to, real_X, real_y), fid_choiceDs_on_real=fid(ts, real_X, real_y))
        rows.append(row)
    res["query_nodes"] = rows
    res["query_summary"] = {k: float(np.nanmean([r[k] for r in rows if r.get(k) is not None])) for k in
                            ("mean_ks_base_vs_sem", "near_boundary_base", "near_boundary_sem", "label_entropy_base", "label_entropy_sem")} if rows else {}
    def _m(key):
        v = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(v)) if v else None
    res["query_summary"].update(
        nodes=len(rows), share_same_choice=_m("same_choice_Do_vs_Ds"),
        fid_choiceDo_on_Do=_m("fid_choiceDo_on_Do"), fid_choiceDo_on_Ds=_m("fid_choiceDo_on_Ds"),
        fid_choiceDs_on_Do=_m("fid_choiceDs_on_Do"), fid_choiceDs_on_Ds=_m("fid_choiceDs_on_Ds"),
        fid_choiceDo_on_real=_m("fid_choiceDo_on_real"), fid_choiceDs_on_real=_m("fid_choiceDs_on_real"),
        share_same_choice_Do_vs_Dalt=_m("same_choice_Do_vs_Dalt"), fid_choiceDalt_on_real=_m("fid_choiceDalt_on_real"),
        fid_choiceDo_on_Dalt=_m("fid_choiceDo_on_Dalt"), fid_choiceDalt_on_Dalt=_m("fid_choiceDalt_on_Dalt"))
    res["tree_split_texts"] = [r.get("test") for r in tree.split_audit_]
    res["derived_literals_in_final_inert_tree"] = int(sum(("onto_" in (t or "")) for t in res["tree_split_texts"]))
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    CAP.clear()
    res = run_diag(a.unit[0], int(a.unit[1]))
    Path(a.out).write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
