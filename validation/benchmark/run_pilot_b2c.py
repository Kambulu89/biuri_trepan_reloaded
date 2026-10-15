"""Piloto DEV de B2-C (EXPERIMENTAL; Doc. 20): igual a ``run.py --unit`` mas com os braços do piloto B2-C acrescentados a A–F, no MESMO split, MLP/FrozenOracle,
ontologia e tuning estrutural ontology-blind. Só datasets DEV. Captura a auditoria B2-C por nó e os tempos por componente de cada árvore.

    python -m validation.benchmark.run_pilot_b2c --unit iris 42 --out DIR
Braços novos (grupo B2C): trepan_original_enr (Original equivalente na representação OWL); b1_tiebreak, b1_calibrated_l0p2, b1_calibrated_l0p4 (sem B2-C);
b2c_{legacy,tiebreak,calibrated_l0p2,calibrated_l0p4} (+ ``_shuf``: controlo com ontologia permutada). O histórico sem B2-C é o braço E (reloaded_owl_full).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

from validation.benchmark import run as R      # PRIMEIRO: fixa o ambiente numérico antes de importar numpy
from core.benchmark import runner as RN
from validation.benchmark import manifest as mf

CAPTURE = {"arm": None, "unit": None, "out": None}
TIMES = ("query_time_", "split_search_time_", "m_of_n_search_time_", "synthetic_generation_time_", "local_model_time_", "oracle_prediction_time_", "training_time_")


def _dump(tree, kind: str) -> None:
    if CAPTURE["out"] is None or CAPTURE["arm"] is None:
        return
    d = Path(CAPTURE["out"]) / "audits"
    d.mkdir(parents=True, exist_ok=True)
    rec = {"kind": kind, "arm": CAPTURE["arm"], "unit": CAPTURE["unit"],
           "timing": {k: float(getattr(tree, k, 0.0) or 0.0) for k in TIMES},
           "b2c_rows": getattr(tree, "b2c_audit_", None), "oracle_accounting": getattr(tree, "oracle_accounting_", None),
           "membership_queries": int(getattr(tree, "membership_queries_", 0)), "nodes_created": getattr(tree, "nodes_created_", None),
           "summary": tree.semantic_audit_summary() if hasattr(tree, "semantic_audit_summary") else None,
           "split_tests": [r.get("test") for r in (getattr(tree, "split_audit_", None) or [])]}
    (d / f"{CAPTURE['unit']}__{CAPTURE['arm']}.json").write_text(json.dumps(rec, default=str), encoding="utf-8")


class _RelCapture(RN.TrepanReloadedClassifier):
    def fit(self, *a, **k):
        r = super().fit(*a, **k)
        _dump(self, "reloaded")
        return r


class _OrigCapture(RN.TrepanOriginalClassifier):
    def fit(self, *a, **k):
        r = super().fit(*a, **k)
        _dump(self, "original")
        return r


def run_unit_pilot(dataset: str, seed: int, out: Path) -> Path:
    CAPTURE.update(unit=f"{dataset}_seed{seed}", out=str(out))
    orig_cfg, orig_groups = R.benchmark_config, mf.MAIN_ARM_GROUPS
    orig_run_arm, orig_rel, orig_orig = RN.BenchmarkRunner._run_arm, RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier

    def wrapped_run_arm(self, spec, *a, **k):
        CAPTURE["arm"] = spec.arm_id
        return orig_run_arm(self, spec, *a, **k)

    R.benchmark_config = lambda s: dataclasses.replace(orig_cfg(s), b2c_pilot=True)
    mf.MAIN_ARM_GROUPS = tuple(orig_groups) + ("B2C",)
    RN.BenchmarkRunner._run_arm = wrapped_run_arm
    RN.TrepanReloadedClassifier, RN.TrepanOriginalClassifier = _RelCapture, _OrigCapture
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
    run_unit_pilot(a.unit[0], int(a.unit[1]), Path(a.out))


if __name__ == "__main__":
    sys.exit(main())
