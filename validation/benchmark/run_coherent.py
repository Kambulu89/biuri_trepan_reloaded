"""Piloto/benchmark DEV da amostragem coerente (EXPERIMENTAL; Doc. 21/22): igual a ``run.py --unit`` + braços do grupo COH + métricas de coerência das consultas
efetivamente geradas em cada árvore. Só datasets DEV. Uma unidade por processo; retoma por checkpoint de tuning.
    python -m validation.benchmark.run_coherent --unit iris 42 --out DIR
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
from validation.benchmark import manifest as mf
from validation.benchmark import coherence_metrics as CM

CAP: dict = {"arm": None, "unit": None, "out": None, "ctx": None, "Xtr": None, "queries": None}
TIMES = ("query_time_", "split_search_time_", "m_of_n_search_time_", "synthetic_generation_time_", "local_model_time_", "oracle_prediction_time_", "training_time_")


def _dump(tree, kind: str) -> None:
    if CAP["out"] is None or CAP["arm"] is None:
        return
    ctx, Xtr = CAP["ctx"], CAP["Xtr"]
    Q = np.vstack(CAP["queries"]) if CAP["queries"] else np.empty((0, 0))
    rec = {"kind": kind, "arm": CAP["arm"], "unit": CAP["unit"], "n_query_rows_captured": int(len(Q)),
           "timing": {k: float(getattr(tree, k, 0.0) or 0.0) for k in TIMES}, "membership_queries": int(getattr(tree, "membership_queries_", 0)),
           "oracle_accounting": getattr(tree, "oracle_accounting_", None), "nodes_created": getattr(tree, "nodes_created_", None),
           "summary": tree.semantic_audit_summary() if hasattr(tree, "semantic_audit_summary") else None,
           "split_tests": [r.get("test") for r in (getattr(tree, "split_audit_", None) or [])],
           "split_features": [sorted({int(l.feature) for l in n.test.literals}) for n in tree.iter_nodes() if getattr(n, "test", None) is not None]}
    if len(Q) and ctx is not None and Xtr is not None and Q.shape[1] == ctx.n_features:
        ref = ctx.enrich(Xtr)
        rec["coherence"] = {**CM.algebraic_violation(Q, ctx.enrich, ctx.orig_idx, ctx.onto_idx, ref), **CM.domain_report(Q, ref, ctx.orig_idx),
                            **CM.distribution_gap(Q, ref, ctx.orig_idx, ctx.onto_idx, seed=0)}
    elif len(Q):
        allc = list(range(Q.shape[1]))
        rec["coherence"] = {**CM.domain_report(Q, Xtr, allc), **CM.distribution_gap(Q, Xtr, allc, [], seed=0), "note": "arm in original space (no derived features)"}
    d = Path(CAP["out"]) / "audits"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{CAP['unit']}__{CAP['arm']}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
    CAP["queries"] = None


def _hook(cls):
    class Cap(cls):
        def fit(self, *a, **k):
            CAP["queries"] = []
            r = super().fit(*a, **k)
            _dump(self, "tree")
            return r

        def _draw_membership_queries(self, model, n, constraints, node):
            out = super()._draw_membership_queries(model, n, constraints, node)
            if CAP["queries"] is not None and len(out):
                CAP["queries"].append(np.asarray(out, dtype=float))
            return out
    return Cap


def run_unit_coherent(dataset: str, seed: int, out: Path) -> Path:
    CAP.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_cfg, orig_groups = R.benchmark_config, mf.MAIN_ARM_GROUPS
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier

    def wrapped_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, *a, **k):
        CAP["arm"] = spec.arm_id
        CAP["Xtr"] = Xtr
        CAP["ctx"] = (ctx_shuf if spec.semantic == "shuffled" else (getattr(self, "_ctx_conv", None) if spec.semantic == "conventional" else ctx_real)) \
            if spec.tree_space == "enr" else None
        return orig_run_arm(self, spec, ds, Xtr, ytr, Xte, yte, Xte_enr, mlp_o, mlp_e, info_o, info_e, ctx_real, ctx_shuf, *a, **k)

    R.benchmark_config = lambda s: dataclasses.replace(orig_cfg(s), coherent_pilot=True)
    mf.MAIN_ARM_GROUPS = tuple(orig_groups) + ("COH",)
    RN.BenchmarkRunner._run_arm = wrapped_run_arm
    RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = _hook(orig_rel), _hook(orig_orig)
    try:
        return R.run_unit(dataset, seed, out)
    finally:
        R.benchmark_config, mf.MAIN_ARM_GROUPS = orig_cfg, orig_groups
        RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = orig_run_arm, orig_rel, orig_orig


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unit", nargs=2, metavar=("DATASET", "SEED"), required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run_unit_coherent(a.unit[0], int(a.unit[1]), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
